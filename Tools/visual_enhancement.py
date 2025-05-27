import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import os
import numpy as np
import time, tqdm
from concurrent.futures import ThreadPoolExecutor

torch.backends.cudnn.benchmark = True
torch.cuda.empty_cache()

def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for i in range(num_basic_block):
        layers.append(basic_block(**kwarg))
        # if i == (num_basic_block // 2) or i == num_basic_block - 1:
        #     layers.append(AttentionBlock(**kwarg))
    return nn.Sequential(*layers)

def get_norm(norm, num_channels, num_groups):
    if norm == "in":
        return nn.InstanceNorm2d(num_channels, affine=True)
    elif norm == "bn":
        return nn.BatchNorm2d(num_channels)
    elif norm == "gn":
        return nn.GroupNorm(num_groups, num_channels)
    elif norm is None:
        return nn.Identity()
    else:
        raise ValueError("unknown normalization type")

class AttentionBlock(nn.Module):
    __doc__ = r"""Applies QKV self-attention with a residual connection.

    Input:
        x: tensor of shape (N, in_channels, H, W)
        norm (string or None): which normalization to use (instance, group, batch, or none). Default: "gn"
        num_groups (int): number of groups used in group normalization. Default: 32
    Output:
        tensor of shape (N, in_channels, H, W)
    Args:
        in_channels (int): number of input channels
    """

    def __init__(self, num_feat, num_grow_ch):
        super().__init__()

        self.in_channels = num_feat
        self.norm = get_norm(None, num_feat, 32)
        self.to_qkv = nn.Conv2d(num_feat, num_feat * 3, 1)
        self.to_out = nn.Conv2d(num_feat, num_feat, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = torch.split(self.to_qkv(self.norm(x)), self.in_channels, dim=1)

        q = q.permute(0, 2, 3, 1).view(b, h * w, c)
        k = k.view(b, c, h * w)
        v = v.permute(0, 2, 3, 1).view(b, h * w, c)

        dot_products = torch.bmm(q, k) * (c ** (-0.5))
        assert dot_products.shape == (b, h * w, h * w)

        attention = torch.softmax(dot_products, dim=-1)
        out = torch.bmm(attention, v)
        assert out.shape == (b, h * w, c)
        out = out.view(b, h, w, c).permute(0, 3, 1, 2)

        return self.to_out(out) + x


class ResidualDenseBlock(nn.Module):
    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        # default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        # print(out.shape)
        out = self.rdb2(out)
        # print(out.shape)
        out = self.rdb3(out)
        # print(out.shape)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x


def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    if hh % scale != 0:
        x = x[:, :, :-(hh%scale), :]
    if hw % scale != 0:
        x = x[:, :, :, :-(hw%scale)]
    b, c, hh, hw = x.size()
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)



class RRDBNet(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(RRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # upsample
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        if scale == 8:
            self.conv_up3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.body(feat)
        body_feat = self.conv_body(body_feat)
        # body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        # upsample
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        # if self.scale == 8:
        #     feat = self.lrelu(self.conv_up3(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


from torch.utils.data import Dataset, DataLoader
class ImageDataset(Dataset):
    def __init__(self, input_folder, img_size):
        self.input_folder = input_folder
        self.img_size = img_size
        self.image_files = []
        # 用进度条tqdm
        for name in tqdm.tqdm(os.listdir(input_folder)):
            if name.endswith(('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp')):
                self.image_files.append(name)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.input_folder, self.image_files[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)

        img = torch.from_numpy(img).float() / 255.0
        img = img.permute(2, 0, 1)
        return img, self.image_files[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visual Enhancement')

    # Data parameters
    parser.add_argument('--input_folder', default=r'F:\Input_IMG_all', help='input path of images that need to be enhanced')
    parser.add_argument('--output_folder', default=r'H:\Output_IMG_all', help='output path of resulted images')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for processing images')
    parser.add_argument('--ckpt_path', default=r'./best_PSNR_iter_22000_new.pth',
                        help='model path of the pretrained model')
    args = parser.parse_args()

    ckpt_path = args.ckpt_path
    batch_size = args.batch_size
    input_folder = args.input_folder
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    state_dict = torch.load(ckpt_path)['params_ema']
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    model = model.to(device)

    img_size = (256, 256)
    os.makedirs(output_folder, exist_ok=True)

    dataset = ImageDataset(input_folder, img_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=8, pin_memory=True)

    print('Processing images...')

    print("TIME:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    # for batch_imgs, batch_img_names in dataloader:
    # 进度条
    for batch_imgs, batch_img_names in tqdm.tqdm(dataloader):
        batch_imgs = batch_imgs.to(device)#.half()  # 转换输入数据为半精度
        with torch.no_grad():
            output_tensor = model(batch_imgs)

        # 后处理
        output_tensor = output_tensor.float().cpu().numpy() * 255
        output_tensor = output_tensor.transpose(0, 2, 3, 1)
        output_tensor = output_tensor.clip(0, 255).round().astype(np.uint8)

        def save_image(img, img_name):
            output = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            output_path = os.path.join(output_folder, img_name)
            cv2.imwrite(output_path, output)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(save_image, output_tensor[i], batch_img_names[i]) for i in range(output_tensor.shape[0])]
            for future in futures:
                future.result()  # 确保所有任务都完成

    print('Done')
    print("Current Time:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))