import os
import numpy as np

from torch.utils.data import Dataset, ConcatDataset
# import elasticdeform as ED

class DatasetGen(Dataset):
    def __init__(self, img_path, trans):
        super(DatasetGen, self).__init__()
        self.img_names = sorted([os.path.join(img_path, image_name) for image_name in os.listdir(img_path)])
        self.trans = trans

    def __getitem__(self, index):     
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
        
        image = image.astype(np.float64)
        image *= 1/image.max()# normalize to 0 and 1
        
        if image.ndim == 2:#expand single channel images
            image = image[np.newaxis, :]
        
        return image, label
    
    def __len__(self):
        return len(self.img_names)

def dataset_gen(img_path, seqs):
    dataset_list = []
    
    if seqs:
        for seq in seqs:
            dataset_list.append(DatasetGen(img_path, seq))
        return ConcatDataset(dataset_list)
    else:
        return DatasetGen(img_path, None)
    
# class DatasetEDGen(DatasetGen):# Elastic deformation
#     def __init__(self, img_path, lbl_path, img_trans, lbl_trans, sigma, points, order):
#         super().__init__(img_path, lbl_path, img_trans, lbl_trans)
#         self.sigma = sigma
#         self.points = points
#         self.order = order
    
#     def __getitem__(self, index):
#         train_image = super().img_standardization(cv2.imread(self.train_names[index],-1))
#         gt_image = super().unit16b2uint8(cv2.imread(self.gt_names[index],-1))
#         gt_image = (np.where(gt_image == 0,0,1)).astype(np.uint8)

#         #ElasticDeformation
#         [train_image, gt_image] = ED.deform_random_grid([train_image, gt_image], 
#                                                 sigma=self.sigma, points=self.points, order=self.order, axis=[(0,1), (0,1)])

#         if self.transform is not None:
#             train_image = self.transform(train_image)

#         return train_image, gt_image