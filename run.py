import os
import re

import torch
import imgaug.augmenters as iaa
##TODO(3)
from torch.utils.data import ConcatDataset, Subset, DataLoader
##TODO(3)
from torch.nn import DataParallel
import numpy as np

from architecture import FeatureNet
from dataset import DatasetGen
from oper import run_model, evaluate
import utils
from metrics import metrics

#TODO(3) config files
#set global variable
os.environ['CUDA_VISIBLE_DEVICES'] = '5,6,7'
seed = 15

#data params
img_path = '/home/yutongx/src_data/images/'
bbox_path = '/home/yutongx/src_data/bbox/'
resize = 32
scale = (0.8,1.2)
translation = (-0.2,0.2)

train_sample_size = 800
train_pos_rate = 0.2
val_sample_size = 200
val_pos_rate = 0.2
# test_sample_size = 1000
# test_pos_rate = 0.5

#training params
epochs = 16
batch_size = 64
test = True

#optim params
lr = 1e-4
betas = (0.9, 0.999)
eps = 1e-08
weight_decay = 1e-4

#lr scheduler params
milestones = [4, 8]
lr_gamma = 0.1

#save params
save_path = './checkpoints/checkpoint_1'

##TODO(3) dataset module
def get_dataset(img_path, bbox_path, label_names, resize=64, augmenter=None):
    
    idx = lambda name: re.sub(r'\D', '', name)
    get_names = lambda path: sorted([os.path.join(path, name) for name in os.listdir(path)], key=idx)
    
    img_names = get_names(img_path)
    bbox_names = get_names(bbox_path)
    
    datasets = []
    for img_name, bbox_name in zip(img_names, bbox_names):
        datasets.append(DatasetGen(img_name, bbox_name, label_names, resize, augmenter))
    
    return ConcatDataset(datasets)

##TODO(3) dataset module
def get_loader(img_path, bbox_path, loader_mode, sample_mode, resize=64, augmenter=None, batch_size=1, 
               sample_size=800, pos_rate=0.2):
    
    assert loader_mode in ['train', 'val', 'test'], f'Invalid mode, got {loader_mode}.'
    assert sample_mode in ['all', 'sampled'], f'Invalid sample mode, got {sample_mode}.'
    
    if loader_mode == 'train':#training set
        img_path = os.path.join(img_path, 'train')
        bbox_path = os.path.join(bbox_path, 'train')
    else:#validation set
        img_path = os.path.join(img_path, 'val')
        bbox_path = os.path.join(bbox_path, 'val')
    
    if sample_mode == 'all':#sample all
        dataset = get_dataset(img_path, bbox_path, label_names = ['gt_pos', 'rpn_pos', 'rpn_neg'], 
                              resize = resize, augmenter = augmenter)
        print(''.join((loader_mode, ' dataset size: ', str(len(dataset)))))
    else:#sampled datatset
        if loader_mode == 'test':#test set
            pos_dataset = get_dataset(img_path, bbox_path, label_names = ['rpn_pos'], resize = resize, augmenter = augmenter)
        else:# train / val set
            pos_dataset = get_dataset(img_path, bbox_path, label_names = ['gt_pos', 'rpn_pos'], 
                                      resize = resize, augmenter = augmenter)
        
        neg_dataset = get_dataset(img_path, bbox_path, label_names = ['rpn_neg'], resize = resize, augmenter = augmenter)
        
        pos_idx = np.random.randint(len(pos_dataset), size = int(sample_size * pos_rate))
        pos_dataset = Subset(pos_dataset, pos_idx)
        
        neg_idx = np.random.randint(len(neg_dataset), size = int(sample_size * (1 - pos_rate)))
        neg_dataset = Subset(neg_dataset, neg_idx)
 
        print(' '.join((loader_mode, sample_mode, 'pos', 'dataset size:', str(len(pos_dataset)))))
        print(' '.join((loader_mode, sample_mode, 'neg', 'dataset size:', str(len(neg_dataset)))))
        
        dataset = ConcatDataset([pos_dataset, neg_dataset])
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def main():
    assert test in [True, False], 'Train/test mode not specified.'
    
    utils.set_global_seed(seed)
    #TODO(2) change to logging
    sys.stdout = utils.Logger(os.path.join(save_path, 'log'))
    
    model = FeatureNet(in_channels=1, out_channels=1)
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
        
    if test == False:
        assert cont in [True, False], 'Scratch/Continuous mode not specified.'
        
        aug = iaa.SomeOf((0, None), [
            iaa.Affine(scale=scale),
            iaa.Affine(translate_percent=translation)])
    
        train_loader = get_loader(img_path, bbox_path, loader_mode='train', sample_mode = 'sampled', resize=resize, augmenter=aug, 
                                  batch_size=batch_size, sample_size=train_sample_size, pos_rate=train_pos_rate)
        val_loader = get_loader(img_path, bbox_path, loader_mode='val', sample_mode = 'sampled', resize=resize, augmenter=aug, 
                                batch_size=batch_size, sample_size=val_sample_size, pos_rate=val_pos_rate)
        if cont == True:
            assert ckpt_path, 'Checkpoint path not found.'
            
            checkpoint = torch.load(ckpt_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optim = torch.optim.Adam(model.parameters())
            optim.load_state_dict(checkpoint['optim_state_dict'])
            
            print(f'Continue Training, starting from epoch {checkpoint['epoch']}')
        else:
            optim = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
            print('Training from scratch.')
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=milestones, gamma=lr_gamma)
        
        run_model(train_loader = train_loader, val_loader = val_loader, model = model, epochs = epochs, 
                  optim = optim, scheduler = scheduler, save_path = save_path)
    else:## change to test.py
        assert ckpt_path, 'Checkpoint path not found.'
        
        test_loader = get_loader(img_path, bbox_path, loader_mode='test', sample_mode = 'all', resize=resize, augmenter=None, 
                                  batch_size=batch_size, sample_size=test_sample_size, pos_rate=test_pos_rate)
        
        checkpoint = torch.load(ckpt_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print('Output Test Results.')
        test_losses, test_y_true, test_y_score = evaluate(test_loader = test_loader, model=model)
        test_loss = np.average(test_losses)
        test_acc, test_prc, test_rec, test_roc_auc, test_curve = metrics(test_y_true, test_y_score, threshold=0.5)
#         TODO(1): change to test.py
#         1. print test results to csv
#         2. print metrics to test log
#         3. print image
#         print('---------------------')
#         print('Test Results:')
#         print('Loss: ', test_loss)
#         print('Accuracy: ', test_acc)
#         print('Precision:', test_prc)
#         print('Recall:', test_rec)
#         print('ROC AUC:', test_roc_auc)
#         plt.imshow(test_curve)
#         plt.show()
#         plt.close()
    
if __name__ == '__main__':
    main()