import os
import sys

import torch
import numpy as np
import imgaug.augmenters as iaa

from architecture import FeatureNet
from dataset.utils import get_dataset, get_loader
from oper import run
import utils

#TODO(3) config files
#set global variable
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
seed = 15

#resume training (if any)
is_cont = False
# ckpt_path = './checkpoints/checkpoint_1/checkpoint.tar.gz'

#data params
img_path = '/home/yutongx/src_data/images/'
bbox_path = '/home/yutongx/src_data/bbox/'

resize = 64
scale = (0.8,1.2)
translation = (-0.2,0.2)
num_workers = 4

train_sample_mode = 'sampled'
train_sample_size = 800
train_pos_rate = 0.5

val_sample_mode = 'sampled'
val_sample_size = 200
val_pos_rate = 0.5

#training params
epochs = 16
batch_size = 64

#optim params
lr = 5e-5
betas = (0.9, 0.999)
eps = 1e-08
weight_decay = 1e-4

#lr scheduler params
milestones = [20, 30]
lr_gamma = 0.1

#save params
save_path = './checkpoints/checkpoint_3'

def main():
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    utils.set_global_seed(seed)
    #TODO(3) change to logging
    sys.stdout = utils.Logger(os.path.join(save_path, 'log'))
    
    model = FeatureNet(in_channels=1, out_channels=1)
    model = utils.gpu_manager(model)

    if is_cont == True:
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim = torch.optim.Adam(model.parameters())
        optim.load_state_dict(checkpoint['optim_state_dict'])
        epoch = checkpoint['epoch']
        print(f'Continue Training, starting from epoch {epoch}.')
    else:
        optim = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        print('Training from scratch.')

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=milestones, gamma=lr_gamma)
    
    aug = iaa.SomeOf((0, None), [
        iaa.Affine(scale=scale),
        iaa.Affine(translate_percent=translation)])

    train_loader = get_loader(img_path, bbox_path, loader_mode='train', sample_mode=train_sample_mode, 
                              resize=resize, augmenter=aug, batch_size=batch_size, 
                              sample_size=train_sample_size, pos_rate=train_pos_rate, num_workers=num_workers)
    
    val_loader = get_loader(img_path, bbox_path, loader_mode='val', sample_mode=val_sample_mode, 
                            resize=resize, augmenter=aug, batch_size=batch_size, 
                            sample_size=val_sample_size, pos_rate=val_pos_rate, num_workers=num_workers)

    run(train_loader=train_loader, val_loader=val_loader, model=model, epochs=epochs, 
              optim=optim, scheduler=scheduler, save_path=save_path, threshold=train_pos_rate)
    
if __name__ == '__main__':
    main()