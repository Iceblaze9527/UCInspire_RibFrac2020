from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
from scipy.ndimage.interpolation import zoom

class DatasetGen(Dataset):
    def __init__(self, img_name, bbox_name, resize=64):
        super(DatasetGen, self).__init__()
        self.image = np.swapaxes(nib.load(img_name).get_fdata(), -1, 0) if isinstance(img_name, str) else raise TypeError('img_name is not a string.')
        self.bboxes = self.get_bbox(bbox_name) if isinstance(bbox_name, str) else raise TypeError('bbox_name is not a string.')
        self.resize = resize if isinstance(self.resize, int) else raise TypeError('resize factor is not an integer.')
        
        assert (self.image).ndim == 3, 'Input dimension mismatch.'
        assert (self.bboxes).shape[1] == 7, 'Bounding box dim mismatch.'

    def __getitem__(self, index): 
        bbox = self.bboxes[index, :-1]
        label = self.bboxes[index, -1]
        
        img = self.crop(self.image, bbox)
        img = zoom(img, [img.shape[0], self.resize, self.resize]) 
        img = np.expand_dims(img, axis=0)
        
        return torch.from_numpy(img), torch.from_numpy(label).long()
    
    def __len__(self):
        return (self.bboxes).shape[0]
    
    @staticmethod
    def get_bbox(bbox_name):
        bboxes_src = np.load(bbox_name, allow_pickle=True)
        
        label_name = ('gt_pos', 'rpn_pos', 'rpn_neg')
        assert set(bboxes_src.files) == set(label_name), 'Label mismatch.'
        
        labelize = lambda name, label: np.concatenate((bboxes_src[name], 
                                                       (label*np.ones((bbox_src[name].shape[0],1))).astype(np.uint8)), axis=1)

        return np.concatenate((labelize(label_name[0],1), labelize(label_name[1],1), labelize(label_name[2],0)), axis=0)
    
    @staticmethod
    def crop(image, bbox):
        zc, yc, xc, dz, dy, dx = bbox
        
        st = {'zs': np.floor(zc - dz/2, dtype=np.uint8)
            'zt': np.ceil(zc + dz/2, dtype=np.uint8) + 1
            'ys': np.floor(yc - dy/2, dtype=np.uint8)
            'yt': np.ceil(yc + dy/2, dtype=np.uint8) + 1
            'xs': np.floor(xc - dx/2, dtype=np.uint8)
            'xt': np.ceil(xc + dx/2, dtype=np.uint8) + 1}
        
        return image[st['zs']:st['zt'], st['ys']:st['yt'], st['xs']:st['xt']]