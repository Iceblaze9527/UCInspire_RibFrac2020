import os
import sys

import torch
import numpy as np
import matplotlib.pyplot as plt

from architecture_4 import FeatureNet
from dataset.utils import get_dataset, get_loader
from oper import evaluate
import utils
from metrics import metrics

#TODO(3) config files
#set global variable
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
seed = 15

#data params
img_path = '/home/yutongx/src_data/images/'
bbox_path = '/home/yutongx/src_data/bbox/'
resize = 64
num_workers = 0

test_sample_mode = 'sampled'
test_sample_size = 2000
test_pos_rate = 0.5

#test params
batch_size = 16

#save params
save_path = './checkpoints/checkpoint_4/'

def main():
    assert os.path.exists(save_path), 'Save path does not exist.'
    
    test_path = os.path.join(save_path, 'test')
    data_path = os.path.join(test_path, 'data')
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    utils.set_global_seed(seed)
    #TODO(3) logger module
    sys.stdout = utils.Logger(os.path.join(test_path, 'log'))
    
    model = FeatureNet(in_channels=1, out_channels=1)  
    model = utils.gpu_manager(model)
    
    checkpoint = torch.load(os.path.join(save_path, 'checkpoint.tar.gz'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loader = get_loader(img_path, bbox_path, loader_mode='test', sample_mode = test_sample_mode, 
                             resize=resize, augmenter=None, batch_size=batch_size, 
                             sample_size=test_sample_size, pos_rate=test_pos_rate, num_workers=num_workers)

    print('Output Test Results.')
    print('====================')
    print('start running at: ', utils.timestamp())
    start = utils.tic()
    
    pos_weight = int((1-test_pos_rate)/test_pos_rate)
    test_losses, test_y_true, test_y_score = evaluate(loader=test_loader, model=model, pos_weight=pos_weight)
    
    print('end running at: ', utils.timestamp())
    end = utils.tic()
    print('overall runtime: ', utils.delta_time(start, end))
    
    test_loss = np.average(test_losses)
    test_acc, test_prc, test_rec, test_roc_auc, test_curve = metrics(
        test_y_true, test_y_score, threshold=test_pos_rate, csv_path=os.path.join(data_path, 'test.csv'))

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