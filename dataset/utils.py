import os
import re

from torch.utils.data import ConcatDataset, Subset, DataLoader
import numpy as np

from dataset.generator import DatasetGen

def get_dataset(img_path, bbox_path, label_names, resize=64, augmenter=None):
    idx = lambda name: re.sub(r'\D', '', name)
    get_names = lambda path: sorted([os.path.join(path, name) for name in os.listdir(path)], key=idx)
    
    img_names = get_names(img_path)
    bbox_names = get_names(bbox_path)
    
    datasets = []
    for img_name, bbox_name in zip(img_names, bbox_names):
        datasets.append(DatasetGen(img_name, bbox_name, label_names, resize, augmenter))
    
    return ConcatDataset(datasets)


def get_loader(img_path, bbox_path, loader_mode, sample_mode, resize=64, augmenter=None, batch_size=1, 
               sample_size=800, num_workers=4):
    
    assert loader_mode in ['train', 'val'], f'Invalid mode, got {loader_mode}.'
    assert sample_mode in ['all', 'sampled'], f'Invalid sample mode, got {sample_mode}.'
    
    idx = lambda dataset, sample_size: np.random.randint(len(dataset), size = int(sample_size))
    
    if loader_mode == 'train':
        img_path = os.path.join(img_path, 'train')
        bbox_path = os.path.join(bbox_path, 'train')
        gt_pos = get_dataset(img_path, bbox_path, label_names = ['gt_pos'], resize = resize, augmenter = augmenter)
    else:
        img_path = os.path.join(img_path, 'val')
        bbox_path = os.path.join(bbox_path, 'val')
    
    rpn_pos = get_dataset(img_path, bbox_path, label_names = ['rpn_pos'], resize = resize, augmenter = augmenter)
    
    pos = ConcatDataset([gt_pos, rpn_pos]) if loader_mode == 'train' else rpn_pos
    pos = Subset(pos, idx(pos, sample_size)) if sample_mode == 'sampled' else pos
    
    print(' '.join((loader_mode, sample_mode, 'dataset size:', str(len(pos)))))
    
    return DataLoader(pos, batch_size=batch_size, shuffle=True, num_workers=num_workers)