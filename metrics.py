import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import pandas as pd

def metrics(results, csv_path, is_multi=False, is_test=False):
    if is_test == False:
        losses, y_name, y_center, y_true, y_score = results
    else:
        y_name, y_center, y_score = results
    
    if is_multi == False:  
        y_pred = np.where(y_score > 0.5, 1, 0).astype(np.uint8)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_score)

        df = pd.DataFrame({'public_id': y_name, 
                           'z_center':y_center[:,0].reshape(-1),
                           'y_center':y_center[:,1].reshape(-1),
                           'x_center':y_center[:,2].reshape(-1),
                           'proba': y_score.reshape(-1),
                           'y_pred': y_pred.reshape(-1)})
    else:
        y_pred = np.argmax(y_score, axis=1)
        
        accuracy = accuracy_score(y_true, y_pred, average='micro')
        precision = precision_score(y_true, y_pred, average='micro')
        recall = recall_score(y_true, y_pred, average='micro')
        roc_auc = roc_auc_score(y_true, y_score, average='micro', multi_class='ovo')
        
        df = pd.DataFrame({'public_id': y_name, 
                   'z_center':y_center[:,0].reshape(-1),
                   'y_center':y_center[:,1].reshape(-1),
                   'x_center':y_center[:,2].reshape(-1),
                   'proba_0': y_score[:,0].reshape(-1),
                   'proba_1': y_score[:,1].reshape(-1),
                   'proba_2': y_score[:,2].reshape(-1),
                   'proba_3': y_score[:,3].reshape(-1),
                   'proba_4': y_score[:,4].reshape(-1),
                   'y_pred': y_pred.reshape(-1)})
    
    df.to_csv(csv_path, index=False, sep=',')
    
    if is_test == False:
        return np.average(losses), accuracy, precision, recall, roc_auc