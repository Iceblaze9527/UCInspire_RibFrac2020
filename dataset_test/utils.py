import os
import re

from torch.utils.data import Subset, ConcatDataset, DataLoader
import numpy as np

from dataset_test.generator import DatasetGen

def get_dataset(img_path, bbox_path, resize=64):
    idx = lambda name: re.sub(r'\D', '', name)
    get_names = lambda path: sorted([os.path.join(path, name) for name in os.listdir(path)], key=idx)
    
    img_names = get_names(img_path)
    bbox_names = get_names(bbox_path)
    
    datasets = []
    for img_name, bbox_name in zip(img_names, bbox_names):
        datasets.append(DatasetGen(img_name, bbox_name, resize))
    
    return ConcatDataset(datasets)


def get_loader(img_path, bbox_path, sample_mode, resize=64, batch_size=1, sample_size=800, num_workers=4):
    assert sample_mode in ['all', 'sampled'], f'Invalid sample mode, got {sample_mode}.'
    
    dataset = get_dataset(img_path, bbox_path, resize = resize)
    
    if sample_mode == 'sampled':
        idx = np.random.randint(len(dataset), size = int(sample_size))
        dataset = Subset(dataset, idx)
    
    print(''.join((sample_mode, ' dataset size: ', str(len(dataset)))))
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)