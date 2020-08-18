import csv

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

#         fig.canvas.draw()
#         curve = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#         curve = curve.reshape(fig.canvas.get_width_height()[::-1] + (3,))

def metrics(y_true, y_score, threshold, csv_path):
    y_pred = np.where(y_score > threshold, 1, 0).astype(np.uint8)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    roc_auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    with open(csv_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['No.', 'y_true', 'y_score', 'y_pred'])
        csv_writer.writerows(np.stack([range(1, y_true.shape[0] + 1), y_true, y_score, y_pred], axis=1))

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr)
    plt.close(fig)
        
    return accuracy, precision, recall, roc_auc, fig