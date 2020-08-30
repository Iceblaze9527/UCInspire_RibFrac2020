import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import pandas as pd

def metrics(results, csv_path, is_test=False):
    pred = lambda score: np.where(score > 0.5, 1, 0).astype(np.uint8)
    
    if is_test == False:
        y_name, y_center, y_score, y_true, losses = results
        y_pred = pred(y_score)

        df = pd.DataFrame({'public_id': y_name, 
                           'z_center': y_center[:,0].reshape(-1),
                           'y_center': y_center[:,1].reshape(-1),
                           'x_center': y_center[:,2].reshape(-1),
                           'proba': y_score.reshape(-1),
                           'y_pred': y_pred.reshape(-1),
                           'y_true': y_true.reshape(-1)})

        df.to_csv(csv_path, index=False, sep=',')

        return np.average(losses), accuracy_score(y_true, y_pred), precision_score(y_true, y_pred), \
    recall_score(y_true, y_pred), roc_auc_score(y_true, y_score)
    
    else:
        y_name, y_center, y_score = results
        y_pred = pred(y_score)

        df = pd.DataFrame({'public_id': y_name, 
                           'z_center': y_center[:,0].reshape(-1),
                           'y_center': y_center[:,1].reshape(-1),
                           'x_center': y_center[:,2].reshape(-1),
                           'proba': y_score.reshape(-1),
                           'y_pred': y_pred.reshape(-1)})

        df.to_csv(csv_path, index=False, sep=',')
