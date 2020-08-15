import os
import re


import torch
from torch.utils.data import ConcatDataset, DataLoader
from torch.nn import DataParallel
import imgaug.augmenters as iaa


from architecture import FeatureNet
from dataset import DatasetGen

#set global variable
os.environ['CUDA_VISIBLE_DEVICES'] = '5,6'
seed = 15

#data params
img_path = '/home/yutongx/src_data/images/'
bbox_path = '/home/yutongx/src_data/bbox/'
resize = 64
scale = (0.8,1.2)
translation = (-0.2,0.2)

#optim params
lr = 1e-3
betas = (0.9, 0.999)
eps = 1e-08
weight_decay = 0

#lr scheduler params

#training params
epochs = 16
batch_size = 64

#checkpoint params
save_path = './checkpoints'

def set_global_seed(seed=15):
    seed = 14
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

def get_dataset(img_path, bbox_path, resize=64, augmenter=None):
    idx = lambda name: re.sub(r'\D', '', name)
    get_names = lambda path: sorted([os.path.join(path, name) for name in os.listdir(path)], key=idx)
    
    img_names = get_names(img_path)
    bbox_names = get_names(bbox_path)
    
    datasets = []
    for img_name, bbox_name in zip(img_names, bbox_names):
        datasets.append(DatasetGen(img_name, bbox_name, resize, augmenter))
    
    return ConcatDataset(datasets)

def get_loader(img_path, bbox_path, mode, resize=64, augmenter=None, batch_size=1):
    img_path = os.path.join(img_path, mode)
    bbox_path = os.path.join(bbox_path, mode)

    dataset = get_dataset(img_path, bbox_path, resize = resize, augmenter = augmenter)
    print(''.join((mode, ' dataset size: ', str(len(dataset)))))
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def main():
    set_global_seed(seed)
    
    aug = iaa.SomeOf((0, None), [
        iaa.Affine(scale=scale),
        iaa.Affine(translate_percent=translation)
    ])
          
    train_loader = get_loader(img_path, bbox_path, 'train', resize=resize, augmenter=aug, batch_size=batch_size)
    val_loader = get_loader(img_path, bbox_path, 'val', resize=resize, augmenter=aug, batch_size=batch_size)
    
    model = FeatureNet(in_channels=1, out_channels=1)
    device_cnt = torch.cuda.device_count()
    if device_cnt > 0:
        if device_cnt == 1:
            print('Only one GPU is available.')
        else:
            print(f"{device_cnt} GPUs available.")
            model = DataParallel(model)
        model = model.cuda()
    else:
        print('Only CPU is available.')
    
#     optim = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

#     ##TODO: lr scheduler
    
#     oper.run_model(train_loader = train_loader, val_loader = val_loader, model = model, optim = optim, 
#                    epochs = epochs, batch_size = batch_size, save_path = save_path)
    
if __name__ == '__main__':
    main()