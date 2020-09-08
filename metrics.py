import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, precision_recall_curve
import pandas as pd

def metrics(results, csv_path, is_test=False):
    pred = lambda score: np.where(score > 0.5, 1, 0).astype(np.uint8)
    
    if is_test == False:
        y_name, y_box, y_score, y_true, losses = results
        y_pred = pred(y_score)

        df = pd.DataFrame({'public_id': y_name, 
                           'z_center': y_box[:,0].reshape(-1),
                           'y_center': y_box[:,1].reshape(-1),
                           'x_center': y_box[:,2].reshape(-1),
                           'z_len': y_box[:,3].reshape(-1),
                           'y_len': y_box[:,4].reshape(-1),
                           'x_len': y_box[:,5].reshape(-1),
                           'proba': y_score.reshape(-1),
                           'y_pred': y_pred.reshape(-1),
                           'y_true': y_true.reshape(-1)})

        df.to_csv(csv_path, index=False, sep=',')

        acc = accuracy_score(y_true, y_pred)
        prc = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0) 
        roc_auc = roc_auc_score(y_true, y_score)
        prc_rec = precision_recall_curve(y_true, y_score)
        
        return np.average(losses), acc, prc, rec, roc_auc, prc_rec
    
    else:
        y_name, y_box, y_score = results
        y_pred = pred(y_score)

        df = pd.DataFrame({'public_id': y_name, 
                           'z_center': y_box[:,0].reshape(-1),
                           'y_center': y_box[:,1].reshape(-1),
                           'x_center': y_box[:,2].reshape(-1),
                           'z_len': y_box[:,3].reshape(-1),
                           'y_len': y_box[:,4].reshape(-1),
                           'x_len': y_box[:,5].reshape(-1),
                           'proba': y_score.reshape(-1),
                           'y_pred': y_pred.reshape(-1)})

        df.to_csv(csv_path, index=False, sep=',')
