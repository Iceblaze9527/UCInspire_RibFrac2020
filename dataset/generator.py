import torch
from torch.utils.data import Dataset

import numpy as np
import nibabel as nib
from scipy.ndimage.interpolation import zoom

class DatasetGen(Dataset):
    def __init__(self, img_name, bbox_name, label_names, resize=64, augmenter=None):
        super(DatasetGen, self).__init__()
        if not isinstance(label_names, list):
            raise TypeError('label_names is not a list.')
        
        self.img_name = img_name
        self.resize = resize
        self.aug = augmenter
        
        self.bboxes = self.get_bboxes(bbox_name, label_names)
        assert (self.bboxes).shape[1] == 7, f'Bounding box dim mismatch, got {(self.bboxes).shape[1]}.'

    def __getitem__(self, index):
        bbox = self.bboxes[index, :-1]
        label = self.bboxes[index, -1] 
        img = nib.load(self.img_name).get_fdata()#H*W*D
        assert img.ndim == 3, f'Input dimension mismatch, , got {img.ndim}.'
        
        img = self.crop(img, bbox, self.resize)#H*W*D
#         length = int((max(img.shape)+1)//2 * 2)#nearest even number
#         img = zoom(img, self.resize/length, order=0)
        
        img = self.aug(image=img) if self.aug is not None else img
        img = np.expand_dims(np.swapaxes(img, -1, 0), axis=0)#H*W*D -> D*H*W -> C*D*H*W
        
        return torch.from_numpy(img), torch.from_numpy(np.array([label]))
    
    def __len__(self):
        return (self.bboxes).shape[0]
    
    @staticmethod
    def get_bboxes(bbox_name, label_names):
        assert set(label_names).issubset(set(['gt_pos', 'rpn_pos', 'rpn_neg'])), f'Label Mismatch, got {label_names}.'
        
        labelize = lambda record, label: np.concatenate((record, label*np.ones((record.shape[0],1)).astype(np.uint8)), axis=1)
        labeled = lambda record, label: labelize(record.reshape(0,6), label) if record.shape[0] == 0 else labelize(record, label)
        
        bboxes_src = np.load(bbox_name, allow_pickle=True)
        
        bbox_data = np.array([]).reshape(0,7)
        for name in label_names:
            if name == 'rpn_neg':
                bbox_data = np.concatenate((bbox_data, labeled(bboxes_src[name], 0)), axis=0)
            else:
                bbox_data = np.concatenate((bbox_data, labeled(bboxes_src[name], 1)), axis=0)
                
        return bbox_data
    
    @staticmethod
    def crop(image, bbox, length):
        start = lambda center, length: int(np.floor(center - length/2))
        end = lambda center, length: int(np.floor(center + length/2))

        start_crop = lambda center, length: int(max(start(center, length), 0))
        end_crop = lambda center, length, max_len: int(min(end(center, length), max_len))

        zc, yc, xc, dz, dy, dx = bbox

        st = {'zs': start_crop(zc, length),
            'zt': end_crop(zc, length, image.shape[2]),
            'ys': start_crop(yc, length),
            'yt': end_crop(yc, length, image.shape[1]),
            'xs': start_crop(xc, length),
            'xt': end_crop(xc, length, image.shape[0])}

        img = image[st['xs']:st['xt'], st['ys']:st['yt'], st['zs']:st['zt']]

        img = np.pad(img, ((abs(start(xc, length)),0),(0,0),(0,0)), 'constant') if start(xc, length) < 0 else img
        img = np.pad(img, ((0,  abs(end(xc, length)) - image.shape[0]),(0,0),(0,0)), 'constant') if end(xc, length) > image.shape[0] else img

        img = np.pad(img, ((0,0),(abs(start(yc, length)),0),(0,0)), 'constant') if start(yc, length) < 0 else img
        img = np.pad(img, ((0,0),(0,  abs(end(yc, length)) - image.shape[1]),(0,0)), 'constant') if end(yc, length) > image.shape[1] else img

        img = np.pad(img, ((0,0),(0,0),(abs(start(zc, length)),0)), 'constant') if start(zc, length) < 0 else img
        img = np.pad(img, ((0,0),(0,0),(0,  abs(end(zc, length)) - image.shape[2])), 'constant') if end(zc, length) > image.shape[2] else img

        return img
        
#     @staticmethod
#     def pad(image, length):
#         start = lambda dim: int(length//2 - dim//2)
#         end = lambda dim: start(dim) + dim
        
#         z, y, x = image.shape
#         canvas = np.zeros((length, length, length))
#         canvas[start(z):end(z), start(y):end(y), start(x):end(x)] = image
        
#         return canvas
        