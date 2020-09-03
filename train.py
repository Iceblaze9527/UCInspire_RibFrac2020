import os
import sys

import torch
import torch.nn as nn
import imgaug.augmenters as iaa

from architecture import FeatureNet
from oper import run
import utils

#TODO(3) config files
#set global variable
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
seed = 15

#training params
is_cont = False #resume training (if any)
#ckpt_path = './checkpoints/checkpoint_3/checkpoint.tar.gz'
epochs = 8

#loader params
img_path = '/home/yutongx/src_data/images/'
bbox_path = '/home/yutongx/src_data/bbox_binary/'
resize = 64
batch_size = 64
num_workers = 8

scale = (0.8,1.2)
translation = (-0.2,0.2)
rotate = (-45, 45)

#train_params
train_sample_mode = 'sampled'
train_sample_size = 16
train_pos_rate = 0.5

#val_params
val_sample_mode = 'sampled'
val_sample_size = 16
val_pos_rate = 0.5

#optim params
lr = 5e-5
betas = (0.9, 0.999)
eps = 1e-08
weight_decay = 1e-3

#lr scheduler params
milestones = [24, 48]
lr_gamma = 0.5

#save params
save_path = './checkpoints/checkpoint_10'

#param dict
loader_params = {
        'img_path': img_path,
        'bbox_path': bbox_path,
        'resize': resize,
        'batch_size': batch_size,
        'num_workers': num_workers
    }

train_params = {
        'sample_mode': train_sample_mode,
        'sample_size': train_sample_size,
        'pos_rate': train_pos_rate
    }

val_params = {
        'sample_size': val_sample_size,
        'sample_mode': val_sample_mode,
        'pos_rate': val_pos_rate
    }

data_params = (loader_params, train_params, val_params)


def main():
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    utils.set_global_seed(seed)
    #TODO(3) change to logging
    sys.stdout = utils.Logger(os.path.join(save_path, 'log'))
    
    model = FeatureNet(in_channels=1, out_channels=1)
    model = utils.gpu_manager(model)
    criterion = nn.BCEWithLogitsLoss(reduction='none')

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
    
    aug = iaa.Sequential([
        iaa.Fliplr(0.5),
        iaa.Flipud(0.5),
        iaa.Affine(scale=scale),
        iaa.Affine(translate_percent=translation),
        iaa.Affine(rotate=rotate)]) 

    run(data_params=data_params, augmenter=aug, model=model, epochs=epochs, optim=optim, 
        criterion=criterion, scheduler=scheduler, save_path=save_path)
    
if __name__ == '__main__':
    main()