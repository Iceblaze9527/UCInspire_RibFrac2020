import torch
from torch.utils.data import Dataset

import numpy as np
import nibabel as nib
from scipy.ndimage.interpolation import zoom

class DatasetGen(Dataset):
    def __init__(self, img_name, bbox_name, label_names, resize=64, augmenter=None):
        super(AllDataset, self).__init__()
        if not isinstance(img_name, str): 
            raise TypeError('img_name is not a string.')
        if not isinstance(bbox_name, str):
            raise TypeError('bbox_name is not a string.')
        if not isinstance(label_names, list):
            raise TypeError('label_names is not a list.')
        if not isinstance(resize, int):
            raise TypeError('resize factor is not an integer.')
        
        self.img_name = img_name
        self.resize = resize
        self.aug = augmenter
        
        self.bboxes = self.get_bboxes(bbox_name, label_names)
        assert (self.bboxes).shape[1] == 7, f'Bounding box dim mismatch, got {(self.bboxes).shape[1]}.'

    def __getitem__(self, index):
        bbox = self.bboxes[index, :-1]
        label = self.bboxes[index, -1] 
        img = np.swapaxes(nib.load(self.img_name).get_fdata(), -1, 0)#H*W*D -> D*H*W
        assert img.ndim == 3, f'Input dimension mismatch, , got {img.ndim}.'
        
        img = self.crop(img, bbox)
        factor = np.array([self.resize, self.resize, self.resize]) / np.array(img.shape)
        img = zoom(img, factor, order=0)
        
        img = self.aug(image=np.swapaxes(img, -1, 0)) if self.aug is not None else np.swapaxes(img, -1, 0) #D*H*W -> H*W*D
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
    def crop(image, bbox):
        start = lambda center, length: int(max(np.floor(center - length/2), 0))
        end = lambda center, length, max_len: int(min(np.ceil(center + length/2), max_len))
        
        zc, yc, xc, dz, dy, dx = bbox
        st = {'zs': start(zc, dz),
            'zt': end(zc, dz, image.shape[0]),
            'ys': start(yc, dy),
            'yt': end(yc, dy, image.shape[1]),
            'xs': start(xc, dx),
            'xt': end(xc, dx, image.shape[2])}

        return image[st['zs']:st['zt'], st['ys']:st['yt'], st['xs']:st['xt']]