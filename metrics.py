import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

#         1. print test results to csv

def metrics(y_true, y_score, threshold):
        y_pred = np.where(y_score > threshold, 1, 0).astype(np.uint8)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        
        roc_auc = roc_auc_score(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
#         fig.canvas.draw()
#         curve = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#         curve = curve.reshape(fig.canvas.get_width_height()[::-1] + (3,))
     
    return accuracy, precision, recall, roc_auc, fig