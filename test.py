import os
import sys

import torch
import numpy as np

from architecture import FeatureNet
from dataset_test.utils import get_loader
from oper import test
import utils
from metrics import metrics

#TODO(3) config files
#set global variable
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
seed = 15

#model params
is_multi = False

#data params
img_path = '/home/yutongx/src_data/images/test/'
bbox_path = '/home/yutongx/src_data/det_bbox_test/'
resize = 64
num_workers = 0

test_sample_mode = 'all'
test_sample_size = 64

#test params
batch_size = 64

#save params
save_path = './checkpoints/checkpoint_9'

def main():
    assert os.path.exists(save_path), 'Save path does not exist.'
    
    test_path = os.path.join(save_path, 'result')
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    
    utils.set_global_seed(seed)
    #TODO(3) logger module
    sys.stdout = utils.Logger(os.path.join(test_path, 'log'))
    
    if is_multi == False:
        model = FeatureNet(in_channels=1, out_channels=1)
    else:
        model = FeatureNet(in_channels=1, out_channels=5)
    
    model = utils.gpu_manager(model)
    
    checkpoint = torch.load(os.path.join(save_path, 'checkpoint.tar.gz'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loader = get_loader(img_path, bbox_path, sample_mode = test_sample_mode, resize=resize,
                             batch_size=batch_size, sample_size=test_sample_size, num_workers=num_workers)

    print('Output Test Results.')
    print('====================')
    print('start running at: ', utils.timestamp())
    start = utils.tic()
    
    test_results = test(loader=test_loader, model=model)
    
    print('end running at: ', utils.timestamp())
    end = utils.tic()
    print('overall runtime: ', utils.delta_time(start, end))

    print('---------------------')
    print(f'Print Results to csv file.')
    
    metrics(test_results, csv_path=os.path.join(test_path, 'result.csv'), is_multi=is_multi, is_test=True)
    
    print('====================')

if __name__ == '__main__':
    main()