<div align="center">
 <h1>
<!--   <img src="images/logo.png" alt="Logo" width="47" height="30" style="margin-right: 10px;"> -->
  <a href="https://ieeexplore.ieee.org/document/10591792">Text2Earth: Unlocking Text-driven Remote Sensing Image Generation with a Global-Scale Dataset and a Foundation Model</a>
</h1>

**[Chenyang Liu](https://chen-yang-liu.github.io/), [Keyan Chen](https://kyanchen.github.io), [Rui Zhao](https://ruizhaocv.github.io/), [Zhengxia Zou](https://scholar.google.com.hk/citations?hl=en&user=DzwoyZsAAAAJ), and [Zhenwei Shi*✉](https://scholar.google.com.hk/citations?hl=en&user=kNhFWQIAAAAJ)**


[![Page](https://img.shields.io/badge/Project-Page-87CEEB)](https://chen-yang-liu.github.io/Text2Earth/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://ieeexplore.ieee.org/document/10988859)
[![YouTube](https://img.shields.io/badge/YouTube-Video-red.svg)](https://youtu.be/Rw9wzUpO01M)

<div align="center">
  <img src="images/overall.png" width="1000"/>
</div>

</div>


## Share us a :star: if you're interested in this repo

This is official repository of the paper: ["**Text2Earth: Unlocking Text-driven Remote Sensing Image Generation with a Global-Scale Dataset and a Foundation Model**"](https://ieeexplore.ieee.org/document/10988859), accepted by **IEEE Geoscience and Remote Sensing Magazine**.


## Latest Updates
✅ 2025-05-27: We are preparing training code and pre-trained model of Text2Earth.

✅ 2025-04-16: The paper has been accepted by **IEEE Geoscience and Remote Sensing Magazine**.

✅ 2025-03-03: Our **Git-RSCLIP** model **available**: [[🤗 Huggingface](https://huggingface.co/lcybuaa/Git-RSCLIP) | [🌊 Modelscope](https://modelscope.cn/models/lcybuaa1111/Git-RSCLIP)]

✅ 2025-02-20: The **Git-10M** dataset is **available**: [[🤗 Huggingface](https://huggingface.co/datasets/lcybuaa/Git-10M) | [🌊 Modelscope](https://modelscope.cn/datasets/lcybuaa1111/Git-10M/)].

✅ 2025-01-01: The paper is **available**.

## Table of Contents
- [🛰️ Git-10M Dataset](#Git-10M-Dataset)
  - [Dataset Download](#Dataset-Download)
  - [Visual Quality Enhancement](#Visual-Quality-Enhancement)
- [🧩 Text2Earth Model](#Text2Earth-Model)
  - [Pre-trained Weights](#Pre-trained-Weights)
  - [Demo](#Demo)
  - [Training](#Training)
  - [Evaluation](#Evaluation)
- [🍀 Git-RSCLIP Model](#Git-RSCLIP-Model)
  - [Model Download](#Git-RSCLIP-Download)
  - [Comparison](#Comparison)
- [✍️ Citation](#Citation)

## 🛰️ Git-10M Dataset 
### Dataset Download 
- The Git-10M dataset is a **global-scale** dataset, consisting of **10.5 million** image-text pairs with geographical locations and resolution information.
- The Git-10M dataset is **available** at: [[🤗 Huggingface](https://huggingface.co/datasets/lcybuaa/Git-10M) | [🌊 Modelscope](https://modelscope.cn/datasets/lcybuaa1111/Git-10M/)].

    <br>
    <div align="center">
      <img src="images/dataset.png" width="800"/>
    </div>
    <br>

### Visual Quality Enhancement
- You can skip the following steps if you have higher visual quality requirements for the image. 
- Some collected images exhibited poor visual quality, such as noise and artifact, which could negatively impact the training of image generation models. 
To address this, you can use an image enhancement model pre-trained on my private high-quality remote sensing dataset to improving the overall image quality.

    **Follow the steps below**:
    <details open>

    **Step 1**: 
    ```python
    cd ./Text2Earth/Tools
    ```
    **Step 2**: Run Python code to process images:
    
    ```python
    python visual_quality_enhancement.py \
        --input_dir /path/to/Git-10M/images \
        --output_dir /path/to/Git-10M/enhanced_images
    ```
    
    </details>



## 🧩 Text2Earth model
### Installation

  <details open>

  **Step 1**: Download or clone the repository.
  ```python
  git clone https://github.com/Chen-Yang-Liu/Text2Earth.git
  cd ./Text2Earth
  ```
  **Step 2**: Create a virtual environment named `Text2Earth_env` and activate it.
  ```python
  conda create -n Text2Earth_env python=3.9
  conda activate Text2Earth_env
  ```

  **Step 3**: Install ``accelerate`` then run ``accelerate config``

  **Step 4**: Our Text2Earth is based on [Diffuser](https://huggingface.co/docs/diffusers/installation). Now install Text2Earth:
  ```python
  cd ./Text2Earth
  pip install -e ".[torch]"
  ```
  </details>

### Pre-trained Weights and Demo
Our pre-trained Text2Earth Download Link : [[🤗 Huggingface](https://huggingface.co/lcybuaa/Text2Earth) | [🌊 Modelscope](https://modelscope.cn/models/lcybuaa1111/Text2Earth)].


### Demo

### 可以轻松迁移到现有的一些StableDiffusion2框架中
Text2Earth可以被认为是遥感StableDiffusion


### Training

### Evaluation


### Experimental Results 
Building on the Git-10M dataset, we developed Text2Earth, a 1.3 billion parameter generative foundation model. Text2Earth excels in resolution-controllable text2image generation and demonstrates robust generalization and flexibility across multiple tasks.

- **Comparison of Text2image models on the previous benchmark dataset (RSICD)**:

  On the previous benchmark dataset RSICD, Text2Earth surpasses the previous models with a significant improvement of +26.23 FID and +20.95% Zero-shot OA metric.
    <br>
    <div align="center">
      <img src="images/RSICD_result.png" width="400"/>
    </div>
    <br>

- **Zero-Shot text2image generation**:
Text2Earth can generate specific image content based on user-free text input, without scene-specific fine-tuning or retraining.
  <br>
  <div align="center">
    <img src="images/zero_result.png" width="800"/>
  </div>
  <div align="center">
    <img src="images/zero_result_2.png" width="800"/>
  </div>
  <br>

- **Unbounded Remote Sensing Scene Construction**:
Using our Text2Earth, users can seamlessly and infinitely generate remote sensing images on a canvas, effectively overcoming the fixed-size limitations of traditional generative models. Text2Earth’s resolution controllability is the key to maintaining visual coherence across the generated scene during the expansion process.
    <br>
    <div align="center">
      <img src="images/unbound.png" width="800"/>
    </div>
    <br>

- **Remote Sensing Image Editing**:
Text2Earth can perform scene modifications based on user-provided text such as replacing or removing geographic features. And it ensures that these modifications are seamlessly integrated with the surrounding areas, maintaining continuity and coherence.
    <br>
    <div align="center">
      <img src="images/edit.png" width="800"/>
    </div>
    <br>

- **Cross-Modal Image Generation**:
Text2Earth can be used for Text-Driven Multi-modal Image Generation, including RGB, SAR, NIR, and PAN images.
    <br>
    <div align="center">
      <img src="images/text2multi-modal.png" width="800"/>
    </div>
    <br>

  Text2Earth also exhibits potential in Image-to-Image Translation, containing cross-modal translation and image enhancement, such as PAN to RGB (PAN2RGB), NIR to RGB (NIR2RGB), PAN to NIR (PAN2NIR), super-resolution, and image dehazing.
  <br>
      <div align="center">
        <img src="images/cross_transf.png" width="800"/>
      </div>
      <br>

## ✍️️ Citation
If you find this paper useful in your research, please consider citing:
```
@ARTICLE{10988859,
  author={Liu, Chenyang and Chen, Keyan and Zhao, Rui and Zou, Zhengxia and Shi, Zhenwei},
  journal={IEEE Geoscience and Remote Sensing Magazine}, 
  title={Text2Earth: Unlocking text-driven remote sensing image generation with a global-scale dataset and a foundation model}, 
  year={2025},
  volume={},
  number={},
  pages={2-23},
  doi={10.1109/MGRS.2025.3560455}}
```

## 📖 License
This repo is distributed under [MIT License](https://github.com/Chen-Yang-Liu/Change-Agent/blob/main/LICENSE.txt). The code can be used for academic purposes only.

[//]: # (## Contact Us)

[//]: # (If you have any other questions❓, please contact us in time 👬)
