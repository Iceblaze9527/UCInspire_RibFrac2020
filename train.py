import os
import sys

import torch
import torch.nn as nn
import imgaug.augmenters as iaa

from architecture import FeatureNet
from dataset.utils import get_loader
from oper import run
import utils

#TODO(3) config files
#set global variable
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
seed = 15

#model params
is_cont = False #resume training (if any)
# ckpt_path = './checkpoints/checkpoint_2/checkpoint.tar.gz'

#data params
img_path = '/home/yutongx/src_data/images/'
bbox_path = '/home/yutongx/src_data/bbox_multi/'

resize = 64
scale = (0.8,1.2)
translation = (-0.2,0.2)
rotate = (-45, 45)
num_workers = 8

train_sample_mode = 'all'
train_sample_size = 16

val_sample_mode = 'all'
val_sample_size = 16

#training params
epochs = 16
batch_size = 64

#optim params
lr = 1e-5
betas = (0.9, 0.999)
eps = 1e-08
weight_decay = 1e-3

#lr scheduler params
milestones = [24, 48]
lr_gamma = 0.5

#save params
save_path = './checkpoints/checkpoint_7'

def main():
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    utils.set_global_seed(seed)
    #TODO(3) change to logging
    sys.stdout = utils.Logger(os.path.join(save_path, 'log'))
    
    model = FeatureNet(in_channels=1, out_channels=4)
    model = utils.gpu_manager(model)
    criterion = nn.CrossEntropyLoss(reduction='none')

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

    train_loader = get_loader(img_path, bbox_path, loader_mode='train', sample_mode=train_sample_mode,
                              resize=resize, augmenter=aug, batch_size=batch_size, 
                              sample_size=train_sample_size, num_workers=num_workers)
    
    val_loader = get_loader(img_path, bbox_path, loader_mode='val', sample_mode=val_sample_mode,
                            resize=resize, augmenter=None, batch_size=batch_size, 
                            sample_size=val_sample_size, num_workers=0)

    run(train_loader=train_loader, val_loader=val_loader, model=model, epochs=epochs, optim=optim, 
        criterion=criterion, scheduler=scheduler, save_path=save_path)
    
if __name__ == '__main__':
    main()