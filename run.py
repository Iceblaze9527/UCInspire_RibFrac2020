import os
import re

import torch
from torch.utils.data import ConcatDataset

from architecture import FeatureNet
from dataset import DatasetGen

img_path = '/home/yutongx/src_data/images/'
bbox_path = '/home/yutongx/src_data/bbox/'

img_train_path = os.path.join(img_path, 'train')
img_val_path = os.path.join(img_path, 'val')

bbox_train_path = os.path.join(bbox_path, 'train')
bbox_val_path = os.path.join(bbox_path, 'val')

def dataset_gen(img_path, bbox_path, resize=64):
    idx = lambda name: re.sub(r'\D', '', name)
    get_names = lambda path: sorted([os.path.join(path, name) for name in os.listdir(path)], key=idx)
    
    img_names = get_names(img_path)
    bbox_names = get_names(bbox_path)
    
    datasets = []
    for img_name, bbox_name in zip(img_names, bbox_names):
        datasets.append(DatasetGen(img_name, bbox_name, resize))
    
    return ConcatDataset(datasets)

train_dataset = dataset_gen(img_train_path, bbox_train_path, resize = 64)
val_dataset = dataset_gen(img_val_path, bbox_val_path, resize = 64)

print('train size:', len(train_dataset))
print('val size:', len(val_dataset))