import os
import re

from torch.utils.data import ConcatDataset, Subset, DataLoader
import numpy as np

from dataset.generator import DatasetGen

def get_dataset(img_path, bbox_path, label_names, is_multi=False, resize=64, augmenter=None):
    idx = lambda name: re.sub(r'\D', '', name)
    get_names = lambda path: sorted([os.path.join(path, name) for name in os.listdir(path)], key=idx)
    
    img_names = get_names(img_path)
    bbox_names = get_names(bbox_path)
    
    datasets = []
    for img_name, bbox_name in zip(img_names, bbox_names):
        datasets.append(DatasetGen(img_name, bbox_name, label_names, is_multi, resize, augmenter))
    
    return ConcatDataset(datasets)


def get_loader(img_path, bbox_path, loader_mode, sample_mode, is_multi=False, resize=64, augmenter=None, batch_size=1, 
               sample_size=800, pos_rate=0.2, num_workers=4):
    
    assert loader_mode in ['train', 'val', 'test'], f'Invalid mode, got {loader_mode}.'
    assert sample_mode in ['all', 'sampled'], f'Invalid sample mode, got {sample_mode}.'
    
    if loader_mode == 'train':
        img_path = os.path.join(img_path, 'train')
        bbox_path = os.path.join(bbox_path, 'train')
    else:
        img_path = os.path.join(img_path, 'val')
        bbox_path = os.path.join(bbox_path, 'val')
    
    if sample_mode == 'all':
        if loader_mode == 'test':
            dataset = get_dataset(img_path, bbox_path, label_names = ['rpn_pos', 'rpn_neg'], is_multi=is_multi,
                                  resize = resize, augmenter = augmenter)
        else:
            dataset = get_dataset(img_path, bbox_path, label_names = ['gt_pos', 'rpn_pos', 'rpn_neg'], is_multi=is_multi,
                                  resize = resize, augmenter = augmenter)
        print(''.join((loader_mode, ' dataset size: ', str(len(dataset)))))
    
    else:
        if loader_mode == 'test':
            pos_dataset = get_dataset(img_path, bbox_path, label_names = ['rpn_pos'], is_multi=is_multi,
                                      resize = resize, augmenter = augmenter)
        else:
            pos_dataset = get_dataset(img_path, bbox_path, label_names = ['gt_pos', 'rpn_pos'], is_multi=is_multi,
                                  resize = resize, augmenter = augmenter)
        
        neg_dataset = get_dataset(img_path, bbox_path, label_names = ['rpn_neg'], is_multi=is_multi, 
                                  resize = resize, augmenter = augmenter)
        
        pos_idx = np.random.randint(len(pos_dataset), size = int(sample_size * pos_rate))
        pos_dataset = Subset(pos_dataset, pos_idx)
        
        neg_idx = np.random.randint(len(neg_dataset), size = int(sample_size * (1 - pos_rate)))
        neg_dataset = Subset(neg_dataset, neg_idx)
 
        print(' '.join((loader_mode, sample_mode, 'pos', 'dataset size:', str(len(pos_dataset)))))
        print(' '.join((loader_mode, sample_mode, 'neg', 'dataset size:', str(len(neg_dataset)))))
        
        dataset = ConcatDataset([pos_dataset, neg_dataset])
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)