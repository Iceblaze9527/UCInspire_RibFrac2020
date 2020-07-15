import os
import numpy as np

from torch.utils.data import Dataset, ConcatDataset

class DatasetGen(Dataset):
    def __init__(self, img_path, trans):
        super(DatasetGen, self).__init__()
        self.img_names = sorted([os.path.join(img_path, image_name) for image_name in os.listdir(img_path)])
        self.trans = trans

    def __getitem__(self, index):
        #TODO(2): uniform data loading preprocessing function
        img_pack = (np.load(self.img_names[index], allow_pickle=True))[()]
        
        if img_pack['image'] is None:
            raise ValueError('Fail to load image at index:%d'%(index))
        else:
            image = img_pack['image']
        
        if img_pack['masks'] is None:
            label = np.zeros(image.shape).astype(np.uint8)
        else:
            label = np.where(img_pack['masks']==0,0,1).astype(np.uint8)
        
        
        if self.trans is not None:
            image = self.trans(images=image)
            label = self.trans(images=label)
        
        ##normalization
        image = image.astype(np.float64)
        image *= 1/image.max()# normalize to 0 and 1
        
        if image.ndim == 2:#expand single channel images
            image = image[np.newaxis, :]
        
        return image, label
    
    def __len__(self):
        return len(self.img_names)

def dataset_gen(img_path, seqs):
    if seqs:
        dataset_list = [DatasetGen(img_path, seq) for seq in seqs]
        return ConcatDataset(dataset_list)
    else:
        return DatasetGen(img_path, None)