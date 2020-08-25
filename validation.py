import os
import sys

import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt

from architecture_gap import FeatureNet
from dataset.utils import get_dataset, get_loader
from oper import evaluate
import utils
from metrics import metrics

#TODO(3) config files
#set global variable
os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'
seed = 15

#model params
is_multi = False

#data params
img_path = '/home/yutongx/src_data/images/'
bbox_path = '/home/yutongx/src_data/bbox_multi/'
resize = 64
num_workers = 0

eval_sample_mode = 'sampled'
eval_sample_size = 500
eval_pos_rate = 0.5

batch_size = 64

#save params
save_path = './checkpoints/checkpoint_8'

def main():
    assert os.path.exists(save_path), 'Save path does not exist.'
    
    eval_path = os.path.join(save_path, 'validation')
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)
    
    utils.set_global_seed(seed)
    #TODO(3) logger module
    sys.stdout = utils.Logger(os.path.join(eval_path, 'log'))
    
    if is_multi == False:
        model = FeatureNet(in_channels=1, out_channels=1)
        criterion = nn.BCEWithLogitsLoss(reduction='none')
    else:
        model = FeatureNet(in_channels=1, out_channels=6)
        criterion = nn.CrossEntropyLoss(reduction='none', ignore_index = -1)
    
    model = utils.gpu_manager(model)
    
    checkpoint = torch.load(os.path.join(save_path, 'checkpoint.tar.gz'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    eval_loader = get_loader(img_path, bbox_path, loader_mode='val', sample_mode = eval_sample_mode, is_multi=is_multi,
                             resize=resize, augmenter=None, batch_size=batch_size, 
                             sample_size=eval_sample_size, pos_rate=eval_pos_rate, num_workers=num_workers)

    print('Output Validation Results.')
    print('====================')
    print('start running at: ', utils.timestamp())
    start = utils.tic()
    
    eval_data = evaluate(loader=eval_loader, model=model, criterion=criterion)
    
    print('end running at: ', utils.timestamp())
    end = utils.tic()
    print('overall runtime: ', utils.delta_time(start, end))
    
    eval_losses, *eval_results = eval_data
    eval_loss = np.average(eval_losses)
    eval_acc, eval_prc, eval_rec, eval_roc_auc, eval_curve = metrics(eval_results,
                                                                     csv_path=os.path.join(data_path, 'eval.csv'))

    print('---------------------')
    print('Validation Results:')
    print('Loss: ', eval_loss)
    print('Accuracy: ', eval_acc)
    print('Precision:', eval_prc)
    print('Recall:', eval_rec)
    print('ROC AUC:', eval_roc_auc)
    print('====================')
    
    eval_curve.savefig(os.path.join(data_path, 'eval.png'))

if __name__ == '__main__':
    main()