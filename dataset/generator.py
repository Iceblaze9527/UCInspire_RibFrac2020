import re

import torch
from torch.utils.data import Dataset

import numpy as np
import nibabel as nib

class DatasetGen(Dataset):
    def __init__(self, img_name, bbox_name, label_names, resize=64, augmenter=None):
        super(DatasetGen, self).__init__()
        self.img_name = img_name
        self.resize = resize
        self.aug = augmenter
        
        self.bboxes = self.get_bboxes(bbox_name, label_names)
        assert (self.bboxes).shape[1] == 7, f'Bounding box dim mismatch, got {(self.bboxes).shape[1]}.'

    def __getitem__(self, index):
        public_id = lambda name: ''.join(('RibFrac', re.sub(r"\D", "", name)))
        
        bbox = self.bboxes[index, :-1]
        
        img = nib.load(self.img_name).get_fdata()#H*W*D
        img = self.crop(img, bbox, self.resize)#H*W*D
        img = self.aug(image=img) if self.aug is not None else img
        img = np.expand_dims(np.swapaxes(img, -1, 0), axis=0)#H*W*D -> D*H*W -> C*D*H*W
        
        label = self.bboxes[index, -1]
        
        return torch.from_numpy(img), [torch.from_numpy(np.array([label - 1]).astype(np.int64)), 
                                       public_id(self.img_name), bbox[:3]]
    
    def __len__(self):
        return (self.bboxes).shape[0]
    
    @staticmethod
    def get_bboxes(bbox_name, label_names):
        assert set(label_names).issubset(set(['gt_pos', 'rpn_pos'])), f'Label Mismatch, got {label_names}.'

        bboxes_src = np.load(bbox_name, allow_pickle=True)
        
        bbox_data = np.array([]).reshape(0,7)
        for name in label_names:
            bboxes =  bboxes_src[name].reshape(0,7) if bboxes_src[name].shape[0] == 0 else bboxes_src[name]
            bbox_data = np.concatenate((bbox_data, bboxes), axis=0)
                
        return bbox_data
    
    
    @staticmethod
    def crop(image, bbox, length):
        start = lambda center, length: int(np.floor(center - length/2))
        end = lambda center, length: int(np.floor(center + length/2))

        start_crop = lambda center, length: int(max(start(center, length), 0))
        end_crop = lambda center, length, max_len: int(min(end(center, length), max_len))
        
        pad = lambda img, pad_size, criterion: np.pad(img, pad_size, 'constant', constant_values=-1024) if criterion else img

        zc, yc, xc, dz, dy, dx = bbox

        st = {'zs': start_crop(zc, length),
            'zt': end_crop(zc, length, image.shape[2]),
            'ys': start_crop(yc, length),
            'yt': end_crop(yc, length, image.shape[1]),
            'xs': start_crop(xc, length),
            'xt': end_crop(xc, length, image.shape[0])}

        img = image[st['xs']:st['xt'], st['ys']:st['yt'], st['zs']:st['zt']]

        img = pad(img, ((abs(start(xc, length)),0),(0,0),(0,0)), start(xc, length)<0)
        img = pad(img, ((0,abs(end(xc, length))-image.shape[0]),(0,0),(0,0)), end(xc, length)>image.shape[0])

        img = pad(img, ((0,0),(abs(start(yc, length)),0),(0,0)), start(yc, length)<0)
        img = pad(img, ((0,0),(0,abs(end(yc, length))-image.shape[1]),(0,0)), end(yc, length)>image.shape[1])

        img = pad(img, ((0,0),(0,0),(abs(start(zc, length)),0)), start(zc, length)<0)
        img = pad(img, ((0,0),(0,0),(0,abs(end(zc, length))-image.shape[2])), end(zc, length)>image.shape[2])

        return img