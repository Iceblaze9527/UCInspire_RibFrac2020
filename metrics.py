import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import pandas as pd

def metrics(results, csv_path, is_test=False):
    maxi = lambda score: np.max(score, axis=1)
    pred = lambda score: np.argmax(score, axis=1)
    
    if is_test == False:
        y_name, y_center, y_score, y_true, losses = results
        y_score_max = maxi(y_score)
        y_pred = pred(y_score)

        df = pd.DataFrame({'public_id': y_name, 
                           'z_center': y_center[:,0].reshape(-1),
                           'y_center': y_center[:,1].reshape(-1),
                           'x_center': y_center[:,2].reshape(-1),
                           'proba_1': y_score[:,0].reshape(-1),
                           'proba_2': y_score[:,1].reshape(-1),
                           'proba_3': y_score[:,2].reshape(-1),
                           'proba_4': y_score[:,3].reshape(-1),
                           'proba_max': y_score_max.reshape(-1),
                           'y_pred': (y_pred + 1).reshape(-1),
                           'y_true': (y_true + 1).reshape(-1)})

        df.to_csv(csv_path, index=False, sep=',')
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', labels=[0,1,2,3], zero_division=0)
        conf_mat = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
        roc_auc = roc_auc_score(y_true.reshape(-1), y_score, average='macro', multi_class='ovo', labels=[0,1,2,3])

        return np.average(losses), accuracy, precision, recall, f1, conf_mat, roc_auc 
    
    else:
        y_name, y_center, y_score = results
        y_score_max = maxi(y_score)
        y_pred = pred(y_score)

        df = pd.DataFrame({'public_id': y_name, 
                           'z_center': y_center[:,0].reshape(-1),
                           'y_center': y_center[:,1].reshape(-1),
                           'x_center': y_center[:,2].reshape(-1),
                           'proba_1': y_score[:,0].reshape(-1),
                           'proba_2': y_score[:,1].reshape(-1),
                           'proba_3': y_score[:,2].reshape(-1),
                           'proba_4': y_score[:,3].reshape(-1),
                           'proba_max': y_score_max.reshape(-1),
                           'y_pred': (y_pred + 1).reshape(-1)})

        df.to_csv(csv_path, index=False, sep=',')
