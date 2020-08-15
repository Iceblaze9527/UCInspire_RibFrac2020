import os
import re

import torch
import imgaug.augmenters as iaa
##TODO(3)
from torch.utils.data import ConcatDataset, DataLoader
##TODO(3)
from torch.nn import DataParallel

from architecture import FeatureNet
from dataset import DatasetGen
from oper import run_model
import utils

#TODO(3) config files
#set global variable
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'
seed = 15

#data params
img_path = '/home/yutongx/src_data/images/'
bbox_path = '/home/yutongx/src_data/bbox/'
resize = 64
scale = (0.8,1.2)
translation = (-0.2,0.2)

#training params
epochs = 256
batch_size = 64

#optim params
lr = 1e-3
betas = (0.9, 0.999)
eps = 1e-08
weight_decay = 0

#lr scheduler params
milestones = [128, 192]
lr_gamma = 0.1

#save params
save_path = './checkpoints/checkpoint_1'

def get_dataset(img_path, bbox_path, resize=64, augmenter=None):##TODO(3) dataset module
    idx = lambda name: re.sub(r'\D', '', name)
    get_names = lambda path: sorted([os.path.join(path, name) for name in os.listdir(path)], key=idx)
    
    img_names = get_names(img_path)
    bbox_names = get_names(bbox_path)
    
    datasets = []
    for img_name, bbox_name in zip(img_names, bbox_names):
        datasets.append(DatasetGen(img_name, bbox_name, resize, augmenter))
    
    return ConcatDataset(datasets)

def get_loader(img_path, bbox_path, mode, resize=64, augmenter=None, batch_size=1):##TODO(3) dataset module
    img_path = os.path.join(img_path, mode)
    bbox_path = os.path.join(bbox_path, mode)

    dataset = get_dataset(img_path, bbox_path, resize = resize, augmenter = augmenter)
    print(''.join((mode, ' dataset size: ', str(len(dataset)))))
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def main():
    utils.set_global_seed(seed)
    
    aug = iaa.SomeOf((0, None), [
        iaa.Affine(scale=scale),
        iaa.Affine(translate_percent=translation)
    ])
          
    train_loader = get_loader(img_path, bbox_path, mode='train', resize=resize, augmenter=aug, batch_size=batch_size)
    val_loader = get_loader(img_path, bbox_path, mode='val', resize=resize, augmenter=aug, batch_size=batch_size)
    
    model = FeatureNet(in_channels=1, out_channels=1)
    ##TODO(2) load checkpoint
    ##TODO(3) utils.gpu_manager
    device_cnt = torch.cuda.device_count()
    if device_cnt > 0:
        if device_cnt == 1:
            print('Only 1 GPU is available.')
        else:
            print(f"{device_cnt} GPUs are available.")
            model = DataParallel(model)
        model = model.cuda()
    else:
        print('Only CPU is available.')
    
    optim = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=milestones, gamma=lr_gamma)
    
    run_model(train_loader = train_loader, val_loader = val_loader, model = model, epochs = epochs, 
              optim = optim, scheduler = scheduler, save_path = save_path)
    
if __name__ == '__main__':
    main()