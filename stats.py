import matplotlib.pyplot as plt
import numpy as np
import csv

def print_data(epochs, stats, stats_path):
    all_train_losses, all_train_jaccards, all_val_losses, all_val_jaccards = stats  
    
    csv_file = open(stats_path + '_data.csv', mode='w')
    csv_writer = csv.writer(csv_file, delimiter=',')
    csv_writer.writerow(['epochs','train_loss','train_jaccard','val_loss','val_jaccard'])

    for idx in range(epochs):
        csv_writer.writerow([idx + 1, all_train_losses[idx], all_train_jaccards[idx], 
                             all_val_losses[idx], all_val_jaccards[idx]])

    csv_file.close()

    x = np.linspace(1, epochs, epochs)

    plt.figure(num=2, figsize=(8,6))
    plt.plot(x, all_train_losses, label='train losses')
    plt.plot(x, all_val_losses, label='val losses')

    plt.xlim((1, epochs))
    plt.ylim((0, 1))

    plt.xticks(np.arange(1, epochs, 1))
    plt.yticks(np.arange(0, 0.006, 0.00005))

    plt.xlabel('epochs')
    plt.ylabel('loss')

    plt.legend()
    plt.grid()
    plt.savefig(stats_path+'_loss.png')
    plt.show()
    
    plt.figure(num=2, figsize=(8,6))
    plt.plot(x, all_train_jaccards, label='train jaccards')
    plt.plot(x, all_val_jaccards, label='val jaccards')

    plt.xlim((1, epochs))
    plt.ylim((0, 1))

    plt.xticks(np.arange(1, epochs, 1))
    plt.yticks(np.arange(0, 0.12, 0.001))

    plt.xlabel('epochs')
    plt.ylabel('jaccard')

    plt.legend()
    plt.grid()

    plt.savefig(stats_path+'_jaccard.png')
    plt.show()
    plt.close()