import pandas as pd

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def metrics(results, threshold, csv_path):
    y_name, y_true, y_score = results
    
    y_pred = np.where(y_score > threshold, 1, 0).astype(np.uint8)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    
    df = pd.DataFrame({'public_id': y_name, 'proba': y_score, 'y_pred': y_pred, 'y_true': y_true})
    df.to_csv(csv_path, index=False, sep=',')

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    plt.close(fig)
        
    return accuracy, precision, recall, roc_auc, fig