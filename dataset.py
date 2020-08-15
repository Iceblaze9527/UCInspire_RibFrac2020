import torch
from torch.utils.data import Dataset

import numpy as np
import nibabel as nib
from scipy.ndimage.interpolation import zoom

class DatasetGen(Dataset):
    def __init__(self, img_name, bbox_name, resize=64, augmenter=None):
        super(DatasetGen, self).__init__()
        if not isinstance(img_name, str): 
            raise TypeError('img_name is not a string.')
        if not isinstance(bbox_name, str):
            raise TypeError('bbox_name is not a string.')
        if not isinstance(resize, int):
            raise TypeError('resize factor is not an integer.')
        
        self.img_name = img_name
        self.bboxes = self.get_bboxes(bbox_name)
        self.resize = resize
        self.aug = augmenter
        
        assert (self.bboxes).shape[1] == 7, 'Bounding box dim mismatch.'

    def __getitem__(self, index):
        bbox = self.bboxes[index, :-1]
        label = self.bboxes[index, -1]
        img = np.swapaxes(nib.load(self.img_name).get_fdata(), -1, 0)
        assert img.ndim == 3, 'Input dimension mismatch.'
        
        img = self.crop(img, bbox)
        factor = np.array([self.resize, self.resize, self.resize]) / np.array(img.shape)
        img = zoom(img, factor, order=0)
        
        img = self.aug(image=np.swapaxes(img, -1, 0)) if self.aug is not None else img #H*W*D
        img = np.expand_dims(np.swapaxes(img, -1, 0), axis=0)#C*D*H*W
        
        return torch.from_numpy(img), torch.from_numpy(np.array([label]))
    
    def __len__(self):
        return (self.bboxes).shape[0]
    
    @staticmethod
    def get_bboxes(bbox_name):
        bboxes_src = np.load(bbox_name, allow_pickle=True)
        label_name = ('gt_pos', 'rpn_pos', 'rpn_neg')
        assert set(bboxes_src.files) == set(label_name), 'Label mismatch.'
        
        labelize = lambda record, label: np.concatenate((record, label*np.ones((record.shape[0],1)).astype(np.uint8)), axis=1)
        labeled = lambda record, label: labelize(record.reshape(0,6), label) if record.shape[0] == 0 else labelize(record, label)
        
        return np.concatenate((labeled(bboxes_src[label_name[0]], 1), 
                               labeled(bboxes_src[label_name[1]], 1),
                               labeled(bboxes_src[label_name[2]], 0)), axis=0)
    
    @staticmethod
    def crop(image, bbox):
        zc, yc, xc, dz, dy, dx = bbox
        
        st = {'zs': np.floor(zc - dz/2).astype(np.int32),
            'zt': np.ceil(zc + dz/2).astype(np.int32) + 1,
            'ys': np.floor(yc - dy/2).astype(np.int32),
            'yt': np.ceil(yc + dy/2).astype(np.int32) + 1,
            'xs': np.floor(xc - dx/2).astype(np.int32),
            'xt': np.ceil(xc + dx/2).astype(np.int32) + 1}
        
        assert st['zs'] > 0 and st['ys'] > 0 and st['xs'] > 0, 'Invalid bounding box start point (negative coordinates).'
        
        return image[st['zs']:st['zt'], st['ys']:st['yt'], st['xs']:st['xt']]