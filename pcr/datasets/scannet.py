"""
ScanNet28 / ScanNet200 Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset

from pcr.utils.logger import get_root_logger
from .builder import DATASETS
from .transform import Compose, TRANSFORMS
from .preprocessing.scannet.meta_data.scannet200_constants import VALID_CLASS_IDS_20, VALID_CLASS_IDS_200
import copy
from pcr.datasets.PointWOLF import PointWOLF


@DATASETS.register_module()
class ScanNetDataset(Dataset):
    class2id = np.array(VALID_CLASS_IDS_20)
    
    def __init__(self,
                 split='train',
                 data_root='data/scannet',
                 transform=None,
                 transform_base=None,
                 transform_pointwolf=None, 
                 transform_at=None,
                 transform_end = None,
                 test_mode=False,
                 test_cfg=None,
                 loop=1):
        super(ScanNetDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.transform_base = Compose(transform_base)
        self.transform_pointwolf = Compose(transform_pointwolf)
        self.transform_at = Compose(transform_at)
        self.transform_end = Compose(transform_end)
        self.loop = loop if not test_mode else 1    # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        self.pointwolf = PointWOLF(w_num_anchor=4, w_sample_type='fps', w_sigma=0.5, w_R_range=10, w_S_range=3, w_T_range=0.25)


        if test_mode:
            self.test_voxelize = TRANSFORMS.build(self.test_cfg.voxelize)
            self.test_crop = TRANSFORMS.build(self.test_cfg.crop) if self.test_cfg.crop else None
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()
        logger = get_root_logger()
        logger.info("Totally {} x {} samples in {} set.".format(len(self.data_list), self.loop, split))

    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.pth"))
        elif isinstance(self.split, list):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.pth"))
        else:
            raise NotImplementedError
        return data_list

    def get_data(self, idx):
        data = torch.load(self.data_list[idx % len(self.data_list)])
        coord = data["coord"]
        color = data["color"]
        normal = data["normal"]
        if "semantic_gt20" in data.keys():
            label = data["semantic_gt20"].reshape([-1])
        else:
            label = np.ones(coord.shape[0]) * 255
        data_dict = dict(coord=coord, normal=normal, color=color, label=label)
        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split(".")[0]

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform_base(data_dict)
        #copy - split to two data_dict
        data_dict_pointwolf = copy.deepcopy(data_dict)
        data_dict_at = copy.deepcopy(data_dict)
        # apply pointwolf and at
        data_dict['coord_pointwolf'] = self.pointwolf(data_dict_pointwolf['coord'])[1]
        data_dict['coord_at'] = self.pointwolf.local_transformaton(data_dict_at['coord'][np.newaxis, ...])[0]
        # apply transformations
        data_dict = self.transform_end(data_dict)
        # data_dict_pointwolf = self.transform_end(data_dict_pointwolf)
        # data_dict_at = self.transform_end(data_dict_at)

        return data_dict

    def prepare_test_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        label = data_dict.pop("label")
        data_dict = self.transform(data_dict)
        data_dict_list = []
        for aug in self.aug_transform:
            data_dict_list.append(
                aug(deepcopy(data_dict))
            )

        input_dict_list = []
        for data in data_dict_list:
            data_part_list = self.test_voxelize(data)
            for data_part in data_part_list:
                if self.test_crop:
                    data_part = self.test_crop(data_part)
                else:
                    data_part = [data_part]
                input_dict_list += data_part

        for i in range(len(input_dict_list)):
            input_dict_list[i] = self.post_transform(input_dict_list[i])
        return input_dict_list, label

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        else:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop


@DATASETS.register_module()
class ScanNet200Dataset(ScanNetDataset):
    class2id = np.array(VALID_CLASS_IDS_200)
    
    def get_data(self, idx):
        data = torch.load(self.data_list[idx % len(self.data_list)])
        coord = data["coord"]
        color = data["color"]
        normal = data["normal"]
        if "semantic_gt200" in data.keys():
            label = data["semantic_gt200"].reshape([-1])
        else:
            label = np.zeros(coord.shape[0])
        data_dict = dict(coord=coord, normal=normal, color=color, label=label)
        return data_dict
