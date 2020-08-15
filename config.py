import os
import random

import numpy as np
import torch

# Set seed and fix backend
seed = 14
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False

BASE = '/home/yxy/disk/Repository/RibFrac/NoduleNet/data/' # make sure you have the ending '/'

data_config = {
    'data_dir': BASE + 'combined',
    # directory for putting all preprocessed results for training to this path
    'preprocessed_data_dir': BASE + 'preprocessed',
    # put annotation downloaded from LIDC to this path
    'annos_dir': BASE + 'annotation/LIDC-XML-only/tcia-lidc-xml',
    # Directory for saving intermediate results
    'ctr_arr_save_dir': BASE + 'annotation/mask_test',
    'mask_save_dir': BASE + 'masks_test',
    'mask_exclude_save_dir': BASE + 'masks_exclude_test',
    
    'roi_names': ['nodule'],
    'crop_size': [128, 128, 128],
    'bbox_border': 8,
    'pad_value': 0,
}

net_config = {
    # Net configuration
    'anchors': get_anchors(bases, aspect_ratios),
    'chanel': 1,
    'crop_size': data_config['crop_size'],
    'stride': 4,
    'max_stride': 16,
    'num_neg': 800,
    'th_neg': 0.02,
    'th_pos_train': 0.5,
    'th_pos_val': 1,
    'num_hard': 3,
    'bound_size': 12,

    'augtype': {'flip': True, 'rotate': True, 'scale': True, 'swap': False},
    'r_rand_crop': 0.,
    'pad_value': 170,
    
}


def lr_shedule(epoch, init_lr=0.01, total=200):
    if epoch <= total * 0.5:
        lr = init_lr
    elif epoch <= total * 0.8:
        lr = 0.1 * init_lr
    else:
        lr = 0.01 * init_lr
    return lr

train_config = {
    'net': 'NoduleNet',
    'batch_size': 16,

    'lr_schedule': lr_shedule,
    'optimizer': 'SGD',
    'momentum': 0.9,
    'weight_decay': 1e-4,

    'epochs': 200,
    'epoch_save': 1,
    'epoch_rcnn': 650,
    'epoch_mask': 800,
    'num_workers': 8,

    'train_set_list': ['split/3_train.csv'],
    'val_set_list': ['split/3_val.csv'],
    'test_set_name': 'split/3_val.csv',
    'label_types': ['mask'],
    'DATA_DIR': data_config['preprocessed_data_dir'],
    'ROOT_DIR': os.getcwd()
}

if train_config['optimizer'] == 'SGD':
    train_config['init_lr'] = 0.01
elif train_config['optimizer'] == 'Adam':
    train_config['init_lr'] = 0.001
elif train_config['optimizer'] == 'RMSprop':
    train_config['init_lr'] = 2e-3

train_config['RESULTS_DIR'] = os.path.join(train_config['ROOT_DIR'], 'results')
train_config['out_dir'] = os.path.join(train_config['RESULTS_DIR'], 'cross_val_test')
train_config['initial_checkpoint'] = None #train_config['out_dir'] + '/model/027.ckpt'

config = dict(data_config, **net_config)
config = dict(config, **train_config)