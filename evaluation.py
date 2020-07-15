import numpy as np
from skimage.filters import threshold_otsu
from sklearn.metrics import jaccard_score

#TODO(2): other evaluation metrics

def binary_iou(pred_eval, y_eval):
    batch_size = y_eval.shape[0]
    jaccard = 0# return sum of this batch
    
    for i in range(batch_size):
        threshold = threshold_otsu(pred_eval[i])
        pred_bin = np.where(pred_eval[i] > threshold, 1, 0).astype(np.uint8)
        jaccard += jaccard_score(y_eval[i].ravel(), pred_bin.ravel())
        
    return jaccard