# Reliability-Adaptive Consistency Regularization for Weakly-Supervised Point Cloud Segmentation


This is an official implementation of the following paper:

- **Reliability-Adaptive Consistency Regularization for Weakly-Supervised Point Cloud Segmentation**   
*Zhonghua Wu\*, Yicheng Wu\*, Guosheng Lin, Jianfei Cai*  
International Journal of Computer Vision (IJCV) 2024  
[ [Arxiv](https://arxiv.org/pdf/2303.05164) ]

## News
- *Jan, 2024*: Initial release our RAC-Net codebase.

## Overview

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Quick Start](#quick-start)
- [Model Zoo](#model-zoo)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)

## Installation

### Requirements
- Ubuntu: 18.04 or higher
- CUDA: 10.2 or higher
- PyTorch: 1.10.0 ~ 1.11.0

### Conda Environment

```bash
conda create -n pcr python=3.8 -y
conda activate pcr
conda install ninja -y
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y
conda install -c anaconda h5py pyyaml -y
conda install -c conda-forge sharedarray tensorboardx yapf addict einops scipy plyfile termcolor timm -y
conda install -c pyg pytorch-cluster pytorch-scatter pytorch-sparse -y
pip install torch-geometric

# spconv (SparseUNet)
# refer https://github.com/traveller59/spconv
pip install spconv-cu113

# Open3D (Visualization)
pip install open3d

# PTv1 & PTv2
cd libs/pointops
python setup.py install
cd ../..

# stratified transformer
pip install torch-points3d

# fix dependence, caused by install torch-points3d 
pip uninstall SharedArray
pip install SharedArray==3.2.1

cd libs/pointops2
python setup.py install
cd ../..

# MinkowskiEngine (SparseUNet)
# refer https://github.com/NVIDIA/MinkowskiEngine

# torchsparse (SPVCNN)
# refer https://github.com/mit-han-lab/torchsparse
# install method without sudo apt install
conda install google-sparsehash -c bioconda
export C_INCLUDE_PATH=${CONDA_PREFIX}/include:$C_INCLUDE_PATH
export CPLUS_INCLUDE_PATH=${CONDA_PREFIX}/include:CPLUS_INCLUDE_PATH
pip install --upgrade git+https://github.com/mit-han-lab/torchsparse.git
```

## Data Preparation

### ScanNet v2

The preprocessing support semantic and instance segmentation for both `ScanNet20` and `ScanNet200`.

- Download the [ScanNet](http://www.scan-net.org/) v2 dataset.
- Download ScanNet 20 points annotations to "./pcr/datasets/preprocessing/scannet/".
- Run preprocessing code for raw ScanNet as follows:

```bash
# RAW_SCANNET_DIR: the directory of downloaded ScanNet v2 raw dataset.
# PROCESSED_SCANNET_DIR: the directory of processed ScanNet dataset (output dir).
python pcr/datasets/preprocessing/scannet/preprocess_scannet.py --dataset_root ${RAW_SCANNET_DIR} --output_root ${PROCESSED_SCANNET_DIR}
```

- Link processed dataset to codebase:
```bash
# PROCESSED_SCANNET_DIR: the directory of processed ScanNet dataset.
mkdir data
ln -s ${RAW_SCANNET_DIR} ${CODEBASE_DIR}/data/scannet
```

## Quick Start

### Training
**Train from scratch.** The training processing is based on configs in `configs` folder. 
The training script will generate an experiment folder in `exp` folder and backup essential code in the experiment folder.
Training config, log, tensorboard and checkpoints will also be saved into the experiment folder during the training process.
```bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
# Script (Recommended)
sh scripts/train.sh -p ${INTERPRETER_PATH} -g ${NUM_GPU} -d ${DATASET_NAME} -c ${CONFIG_NAME} -n ${EXP_NAME}
# Direct
export PYTHONPATH=./
python tools/train.py --config-file ${CONFIG_PATH} --num-gpus ${NUM_GPU} --options save_path=${SAVE_PATH}
```

For example:
```bash
# By script (Recommended)
# -p is default set as python and can be ignored
sh scripts/train.sh -p python -d scannet -c semseg-ptv1-0-base -n semseg-ptv1-0-base
# Direct
export PYTHONPATH=./
python tools/train.py --config-file configs/scannet/semseg-ptv1-0-base.py --options save_path=exp/scannet/semseg-ptv1-0-base
```

### Testing
The validation during training only evaluate model on point clouds after grid sampling (voxelization) and testing is need to achieve a precise evaluate result.
Our testing code support TTA (test time augmentation) testing.
(Currently only support testing on a single GPU, I might add support to multi-gpus testing in the future version.)

```bash
# By script (Based on experiment folder created by training script)
sh scripts/test.sh -p ${INTERPRETER_PATH} -d ${DATASET_NAME} -n ${EXP_NAME} -w ${CHECKPOINT_NAME}
# Direct
export PYTHONPATH=./
python tools/test.py --config-file ${CONFIG_PATH} --options save_path=${SAVE_PATH} weight=${CHECKPOINT_PATH}
```

For example:
```bash
# By script (Based on experiment folder created by training script)
# -p is default set as python and can be ignored
# -w is default set as model_best and can be ignored
sh scripts/test.sh -p python -d scannet -n semseg-ptv1-0-base -w model_best
# Direct
export PYTHONPATH=./
python tools/test.py --config-file configs/scannet/semseg-ptv1-0-base.py --options save_path=exp/scannet/semseg-ptv1-0-base weight=exp/scannet/semseg-ptv1-0-base/models/model_best.pth
```

## Model Zoo

Coming soon...

## Citation
If you find this work useful to your research, please cite our work:
```
@article{wu2024reliability,
  title={Reliability-Adaptive Consistency Regularization for Weakly-Supervised Point Cloud Segmentation},
  author={Wu, Zhonghua and Wu, Yicheng and Lin, Guosheng and Cai, Jianfei},
  journal={International Journal of Computer Vision},
  pages={1--14},
  year={2024},
  publisher={Springer}
}
```

## Acknowledgement
The repo is derived from Point Transformer code and inspirited by several repos, e.g., [PointTransformer](https://github.com/Pointcept/PointTransformerV2), [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine), [pointnet2](https://github.com/charlesq34/pointnet2), [mmcv](https://github.com/open-mmlab/mmcv/tree/master/mmcv), and [Detectron2](https://github.com/facebookresearch/detectron2).
