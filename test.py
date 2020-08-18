import os
import sys

import torch
##TODO(3)
from torch.nn import DataParallel
import numpy as np
import matplotlib.pyplot as plt

from architecture import FeatureNet
from dataset.utils import get_dataset, get_loader
from oper import evaluate
import utils
from metrics import metrics

#TODO(3) config files
#set global variable
os.environ['CUDA_VISIBLE_DEVICES'] = '6,7'
seed = 15

#data params
img_path = '/home/yutongx/src_data/images/'
bbox_path = '/home/yutongx/src_data/bbox/'
resize = 32
test_sample_size = 1000
test_pos_rate = 0.5

#test params
batch_size = 64

#save params
save_path = './checkpoints/checkpoint_1/'

def main():
    assert os.path.exists(save_path), 'Save path does not exist.'
    
    test_path = os.path.join(save_path, 'test')
    data_path = os.path.join(test_path, 'data')
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    utils.set_global_seed(seed)
    #TODO(2) change to logging
    sys.stdout = utils.Logger(os.path.join(test_path, 'log'))
    
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
    
    checkpoint = torch.load(os.path.join(save_path, 'checkpoint.tar.gz'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loader = get_loader(img_path, bbox_path, loader_mode='test', sample_mode = 'all', resize=resize, augmenter=None, 
                              batch_size=batch_size, sample_size=test_sample_size, pos_rate=test_pos_rate)

    print('Output Test Results.')
    print('====================')
    print('start running at: ', utils.timestamp())
    start = utils.tic()
    
    test_losses, test_y_true, test_y_score = evaluate(loader=test_loader, model=model)
    
    print('end running at: ', utils.timestamp())
    end = utils.tic()
    print('overall runtime: ', utils.delta_time(start, end))
    
    test_loss = np.average(test_losses)
    test_acc, test_prc, test_rec, test_roc_auc, test_curve = metrics(
        test_y_true, test_y_score, threshold=0.5, csv_path=os.path.join(data_path, 'test.csv'))

    print('---------------------')
    print('Test Results:')
    print('Loss: ', test_loss)
    print('Accuracy: ', test_acc)
    print('Precision:', test_prc)
    print('Recall:', test_rec)
    print('ROC AUC:', test_roc_auc)
    print('====================')
    
    test_curve.savefig(os.path.join(data_path, 'test.png'))

if __name__ == '__main__':
    main()