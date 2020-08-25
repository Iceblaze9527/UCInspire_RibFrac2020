import os
import sys

import torch
import numpy as np
import pandas as pd

from architecture import FeatureNet
from dataset_test.utils import get_dataset, get_loader
from oper import test
import utils

#TODO(3) config files
#set global variable
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
seed = 15

#model params
is_multi = False

#data params
img_path = '/home/yutongx/src_data/images/test/'
bbox_path = '/home/yutongx/src_data/det_bbox_all/'
resize = 64
num_workers = 0

test_sample_mode = 'all'
test_sample_size = 64
threshold = 0.5

#test params
batch_size = 128

#save params
save_path = '/home/yxy/disk/Repository/UCInspire_RibFrac2020/checkpoints/checkpoint_8'
out_path = './checkpoints/checkpoint_8_yxy'

def main():
    assert os.path.exists(save_path), 'Save path does not exist.'
    
    test_path = os.path.join(out_path, 'result')
    csv_path = os.path.join(test_path, 'result.csv')
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    
    utils.set_global_seed(seed)
    #TODO(3) logger module
    sys.stdout = utils.Logger(os.path.join(test_path, 'log'))
    
    if is_multi == False:
        model = FeatureNet(in_channels=1, out_channels=1)
    else:
        model = FeatureNet(in_channels=1, out_channels=6)
    
    model = utils.gpu_manager(model)
    
    checkpoint = torch.load(os.path.join(save_path, 'checkpoint.tar.gz'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loader = get_loader(img_path, bbox_path, sample_mode = test_sample_mode, resize=resize,
                             batch_size=batch_size, sample_size=test_sample_size, num_workers=num_workers)

    print('Output Test Results.')
    print('====================')
    print('start running at: ', utils.timestamp())
    start = utils.tic()
    
    y_name, y_score = test(loader=test_loader, model=model)
    
    print('end running at: ', utils.timestamp())
    end = utils.tic()
    print('overall runtime: ', utils.delta_time(start, end))

    print('---------------------')
    print(f'Print Results to {csv_path}.')
    
    y_pred = np.where(y_score > threshold, 1, 0).astype(np.uint8)
    df = pd.DataFrame({'public_id': y_name, 'proba': y_score.reshape(-1), 'y_pred': y_pred.reshape(-1)})
    df.to_csv(csv_path, index=False, sep=',')
    
    print('====================')

if __name__ == '__main__':
    main()