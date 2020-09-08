import os
import sys

import torch
import torch.nn as nn

from architecture import FeatureNet
from dataset.utils import get_loader
from oper import evaluate
import utils
from metrics import metrics

#TODO(3) config files
#set global variable
os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'
seed = 15

#data params
img_path = '/home/yutongx/src_data/images/'
bbox_path = '/home/yutongx/src_data/bbox_binary/'
resize = 64
batch_size = 16

#eval params
eval_sample_mode = 'all'
eval_sample_size = 16
eval_pos_rate = 0.5

#save params
save_path = './checkpoints/checkpoint_11'
out_path = './checkpoints/checkpoint_11'

#param dict
loader_params = {
        'img_path': img_path,
        'bbox_path': bbox_path,
        'resize': resize,
        'batch_size': batch_size,
        'num_workers': 0
    }

eval_params = {
        'sample_size': eval_sample_size,
        'sample_mode': eval_sample_mode,
        'pos_rate': eval_pos_rate
    }

def main():
    assert os.path.exists(save_path), 'Save path does not exist.'
    
    eval_path = os.path.join(out_path, 'validation')
    if not os.path.exists(eval_path):
        os.makedirs(eval_path)
    
    utils.set_global_seed(seed)
    #TODO(3) logger module
    sys.stdout = utils.Logger(os.path.join(eval_path, 'log'))
    
    model = FeatureNet(in_channels=1, out_channels=1)
    model = utils.gpu_manager(model)
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    checkpoint = torch.load(os.path.join(save_path, 'checkpoint.tar.gz'))
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print(f'Best epoch {epoch}.')
    
    eval_loader = get_loader(loader_mode='val', augmenter=None, **loader_params, **eval_params)

    print('Output Validation Results.')
    print('====================')
    print('start running at: ', utils.timestamp())
    start = utils.tic()
    
    eval_results = evaluate(loader=eval_loader, model=model, criterion=criterion)
    
    print('end running at: ', utils.timestamp())
    end = utils.tic()
    print('overall runtime: ', utils.delta_time(start, end))
    
    eval_loss, eval_acc, eval_prc, eval_rec, eval_roc_auc, eval_prc_rec = metrics(
        eval_results, csv_path=os.path.join(eval_path, 'eval.csv'), is_test=False)
    
    print('---------------------')
    print('Validation Results:')
    print('Loss: ', eval_loss)
    print('Accuracy: ', eval_acc)
    print('Precision:', eval_prc)
    print('Recall:', eval_rec)
    print('ROC AUC:', eval_roc_auc)
    print('====================')
    
    utils.draw_curve(eval_prc_rec, graph_path=os.path.join(eval_path, 'eval.png'))

if __name__ == '__main__':
    main()