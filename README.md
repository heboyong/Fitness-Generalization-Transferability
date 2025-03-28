<!-- PROJECT LOGO -->
<p align="center">

  <h3 align="center">Boosting Domain Generalized and Adaptive Detection with Diffusion Models: Fitness, Generalization, and Transferability</h3>

</p>

## <a id="table-of-contents"></a> 📚 Table of contents

- [ 📚 Table of contents](#--table-of-contents)
- [ 🤗 !!! Trained Modles !!!](#---trained-modles-)
- [ 🌟 Introduction](#--introduction)
- [ 💾 Dataset Prepare](#--dataset-prepare)
  - [Dataset access](#dataset-access)
  - [DG and DA Benchmark](#dg-and-da-benchmark)
- [ 💡 Environment Prepare](#--environment-prepare)
  - [Requirements](#requirements)
  - [Environment Installation](#environment-installation)
- [ 📁 Project Structure](#--project-structure)
- [ 🚀 Training and Testing](#--training-and-testing)
    - [Multi-gpu Training](#multi-gpu-training)
    - [Model Testing](#model-testing)
- [ 🙏 Acknowledgments](#--acknowledgments)
- [ 🎫 License](#--license)

## <a id="Trained Modles"></a> 🤗 !!! Trained Modles !!!
We provide all trained models here [google drive](https://drive.google.com/drive/folders/1e-tAdw_g5rnggAL9tWlwGe_F5LSGcL0g?usp=drive_link). You can find all the weights corresponding to the settings in the models folder, then directly test to get the results as here [Model Testing](#model-testing).

## <a id="Introduction"></a> 🌟 Introduction
This repository is the code implementation of the paper ***Boosting Domain Generalized and Adaptive Detection with Diffusion Models: Fitness, Generalization, and Transferability***, which is based on the [MMDetection](https://github.com/open-mmlab/mmdetection) project, which we focus **domain generalized** and **adaptive** detection with **diffusion models** as:

***Fitness:*** Extracting intermediate features from a single-step diffusion process, optimizing feature collection and fusion mechanisms to reduce inference time by 75% while improving source domain performance.
***Generalization:*** Constructing an object-centered auxiliary branch using masked images with class prompts to help diffusion detectors obtain more robust domain-invariant features focused on objects, and applying consistency loss to align both branches, balancing fitness and generalization.
***Generalization:*** Within a unified framework, guiding standard detectors with diffusion detectors through feature-level and object-level alignment on source domains (for DG) and unlabeled target domains (for DA) to improve cross-domain detection performance.

Our method achieves competitive results on 3 DA benchmarks and 5 DG benchmarks. Extended experiments on COCO demonstrate that our approach maintains significant advantages over stronger models across different data scales, showing remarkable efficiency particularly in scenarios with large domain shifts and limited training data. This work proves the superiority of applying diffusion models to domain generalized and adaptive detection tasks, offering valuable insights for visual perception tasks requiring generalization and adaptation capabilities.

## <a id="dataset-prepare"></a> 💾 Dataset Prepare
### Dataset access
- Image and annotation download link: [Cityscapes, FoggyCityscapes](https://www.cityscapes-dataset.com).
- Image and annotation download link: [BDD 100k](https://bdd-data.berkeley.edu/).
- Image and annotation download link: [VOC 07+12](http://host.robots.ox.ac.uk/pascal/VOC/).
- Image and annotation download link: [Clipart, Comic, Watercolor](https://github.com/naoto0804/cross-domain-detection/tree/master/datasets).
- Image and annotation download link: [Diverse Weather Benchmark](https://github.com/AmingWu/Single-DGOD).

### DG and DA Benchmark

- **Cross Camera.** Train on Cityscapes (2,975 training images from 50 cities) and test on BDD100K day-clear split with 7 shared categories, evaluating generalization across diverse urban scenes.

- **Adverse Weather.** Train on Cityscapes and test on FoggyCityscapes, using the challenging 0.02 split setting for FoggyCityscapes to evaluate robustness under degraded visibility conditions.

- **Real to Artistic.** Train on VOC (16,551 real-world images from 2007 and 2012) and test on Clipart (1K images, 20 categories), Comic (2K images, 6 categories), and Watercolor (2K images, 6 categories).

- **Diverse Weather Benchmark.** Train on Daytime-Sunny (26,518 images) and test on four challenging conditions: Night-Sunny (26,158 images), Night-Rainy (2,494 images), Dusk-Rainy (3,501 images), and Daytime-Foggy (3,775 images), evaluating robustness across diverse weather and lighting scenarios.

- **Corruption Benchmark.** A comprehensive test-only benchmark with 15 different corruption types at 5 severity levels for Cityscapes, spanning noise, blur, weather, and digital perturbations to evaluate model robustness systematically. 

We test DG on all five benchmarks and test DA on **Cross Camera**, **Adverse Weather**, **Real to Artistic** benchmarks.

## <a id="Environment Prepare"></a> 💡 Environment Prepare

### Requirements
- Linux system, Windows is not tested
- Python 3.8+, recommended 3.10
- PyTorch 2.1 or higher, recommended 2.1.0
- CUDA 11.8 or higher, recommended 12.1
- MMCV 2.0 or higher, recommended 2.1.0
- MMDetection 3.0 or higher, recommended 3.3.0
- diffusers 0.30.0 or higher, recommended 0.30.0
### Environment Installation

***Note:*** It is recommended to use conda for installation. The following commands will create a virtual environment named `Diffusion_Detection` and install PyTorch and MMCV. In the following installation steps, the default installed CUDA version is **12.1**. 
If your CUDA version is not 12.1, please modify it according to the actual situation.

**Step 1**: Create a virtual environment named `Diffusion_Detection` and activate it.

```shell
conda create -n Diffusion_Detection python=3.10 -y
conda activate Diffusion_Detection
```

**Step 2**: Install [PyTorch2.x](https://pytorch.org/get-started/locally/).

**Step 3**: Install [MMDetection-3.x](https://mmdetection.readthedocs.io/en/latest/get_started.html).

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.1.0"
mim install mmdet=3.3.0
```

**Step 4**: Prepare for [Stable-diffusion-1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5) with diffusers.

```shell
git lfs install
git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/runwayml/stable-diffusion-v1-5
```
And you should move the **Stable-diffusion-1.5** (SD-1.5) to the same dir as our **Diffusion_Detection**. Then:

```shell
pip install diffusers==0.30.0
```
The configuration steps for (***SD-2.1***, ***SD-3.5-M***) follow the same procedure as previously described.

## <a id="project-structure"></a> 📁 Project Structure

- **DG Experiment Settings:** `DG/`
- **DG Experiment Settings:** `DA/`
- **Training Settings:** `DG/_base_/da_setting/` and `DA/_base_/da_setting/`
- **Dataset Settings:** `DG/_base_/datasets/` and `DA/_base_/datasets/`
- **Model Settings:** `DG/Ours/cityscapes/` and `DA/Ours/city_to_foggy/`
- **Diffusion Backbone:** `mmdet/models/backbones/diff/` and [mmdet/models/backbones/diff_encoder.py](mmdet/models/backbones/diff_encoder.py)
- **Diffusion Detector Code:** [mmdet/models/detectors/Z_diffusion_detector.py](mmdet/models/detectors/Z_diffusion_detector.py)
- **Diffusion Guided Generalization Detector:** [mmdet/models/detectors/Z_domain_generalization_detector.py](mmdet/models/detectors/Z_domain_generalization_detector.py)
- **Diffusion Guided Adaptation Detector:** [mmdet/models/detectors/Z_domain_adaptation_detector.py](mmdet/models/detectors/Z_domain_adaptation_detector.py)
- **Domain Aug. Code:** [mmdet/datasets/transforms/albu_domain_adaption.py](mmdet/datasets/transforms/albu_domain_adaption.py)

## <a id="train-and-test"></a> 🚀 Training and Testing

The models are trained for 20,000 steps on two 4090 GPUs, with a batch size of 16 (For Diverse Weather Benchmark, we use eight 4090 GPUs with a total batch size of 16). 
If your settings are different from ours, please modify the training steps and default learning rate settings in [training config](DG/_base_/dg_setting).


#### Multi-gpu Training
```shell
sh ./tools/dist_train.sh ${CHECKPOINT_FILE} ${GPU_NUM}  # CHECKPOINT_FILE is the configuration file you want to use, GPU_NUM is the number of GPUs used
```
For ***Diffusion Detector*** :
```shell
sh ./tools/dist_train.sh DG/Ours/cityscapes/diffusion_detector_cityscapes.py  2  
```

For ***Diffusion Guided Generalization Detector***  and  *Note* : you should check and set the *detector.diff_model.config* and *detector.diff_model.pretrained_model* correctly in [diffusion_guided_generalization_faster-rcnn_r101_fpn_cityscapes.py](DG/Ours/cityscapes/diffusion_guided_generalization_faster-rcnn_r101_fpn_cityscapes.py).

```shell
sh ./tools/dist_train.sh DG/Ours/cityscapes/diffusion_guided_generalization_faster-rcnn_r101_fpn_cityscapes.py  2  
```

For ***Diffusion Guided Adaptation Detector***  and  *Note* : you should check and set the *detector.diff_model.config* and *detector.diff_model.pretrained_model* correctly in [diffuison_guided_adaptation_faster-rcnn_r101_fpn_city_to_foggy.py](DA/Ours/city_to_foggy/diffuison_guided_adaptation_faster-rcnn_r101_fpn_city_to_foggy.py).

```shell
sh ./tools/dist_train.sh DA/Ours/city_to_foggy/diffuison_guided_adaptation_faster-rcnn_r101_fpn_city_to_foggy.py  2  
```

For ***COCO Generalization Benchmark*** , please see here `configs/diff/`

#### Model Testing

We provide a convenient way to quickly perform DG testing.

```shell
sh ./tools/dist_test_dg.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM}  # CONFIG_FILE is the configuration file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use, GPU_NUM is the number of GPUs used
```

For ***Diffusion Guided Generalization Detector*** , please change the code [here](DG/Ours/cityscapes/diffusion_guided_generalization_faster-rcnn_r101_fpn_cityscapes.py) *detector.dift_model.config* and *detector.dift_model.pretrained_model* as *None* before test, to prevent applying settings and weights related to diffusion models.

```shell
sh ./tools/dist_test_dg.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM}  # CONFIG_FILE is the configuration file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use, GPU_NUM is the number of GPUs used
```

Also , for ***Diffusion Guided Adaptation Detector*** , please change the code [here](DA/Ours/city_to_foggy/diffuison_guided_adaptation_faster-rcnn_r101_fpn_city_to_foggy.py) *detector.dift_model.config* and *detector.dift_model.pretrained_model* as *None* before test, to prevent applying settings and weights related to diffusion models.

```shell
sh ./tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM}  # CONFIG_FILE is the configuration file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use, GPU_NUM is the number of GPUs used
```

For ***COCO Generalization Benchmark*** , please see config here `configs/diff` and tested with:

```shell
sh ./tools/dist_test_dg_coco.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM}  # CONFIG_FILE is the configuration file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use, GPU_NUM is the number of GPUs used
```
## <a id="acknowledgments"></a> 🙏 Acknowledgments 
This work draws inspiration from the following code and settings as references. We extend our gratitude to these remarkable contributions:

- [GDD](https://github.com/heboyong/Generalized-Diffusion-Detector)
- [MMdetection](https://github.com/open-mmlab/mmdetection)
- [DIFF](https://github.com/Yux1angJi/DIFF)
- [Diffusion-HyperFeature](https://github.com/Flamm64/GTA-V-World-Map)
- [OA-DG](https://github.com/WoojuLee24/OA-DG)
- [DivAlign](https://github.com/msohaildanish/DivAlign)


## <a id="license"></a> 🎫 License
This project is licensed under the [Apache 2.0 license](LICENSE).

