import os
import sys

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from tqdm import tqdm

import utils

th = 0.5

def train(loader, model, optim):
    model.train()
    
    losses = np.array([])
    y_true_all = np.array([])
    y_score_all = np.array([])
    
    for idx, (X, y) in tqdm(enumerate(loader), total=len(loader), desc='Training'):
        optim.zero_grad()
        X = X.float().cuda() # [N, IC=1, D, H, W]
        y = y.long().cuda() # foreground  = 1
        pred = model(X) # foreground logit proba [N, 1]
        
        loss = nn.BCEWithLogitsLoss()(pred, y)
        y_true = y.detach().cpu().numpy()
        y_score = torch.sigmoid(pred).detach().cpu().numpy()
        
        losses = np.concatenate((losses, loss.detach().cpu().numpy().reshape(-1)))
        y_true_all = np.concatenate((y_true_all, y_true.reshape(-1)))
        y_score_all = np.concatenate((y_score_all, y_score.reshape(-1)))
        
        loss.backward()
        optim.step()

        del X, y, pred, loss
        torch.cuda.empty_cache()
    
    y_pred_all = np.where(y_score_all > th, 1, 0).astype(np.uint8)
    accuracy = accuracy_score(y_true_all, y_pred_all)
    precision = precision_score(y_true_all, y_pred_all)
    recall = recall_score(y_true_all, y_pred_all)
    
    return np.average(losses), accuracy, precision, recall


def evaluate(loader, model):
    model.eval()

    losses = np.array([])
    y_true_all = np.array([])
    y_score_all = np.array([])
    
    with torch.no_grad():
        for idx, (X, y) in tqdm(enumerate(loader), total=len(loader), desc='Validating'):
            X = X.float().cuda() # [N, IC=1, D, H, W]
            y = y.float().cuda() # foreground=1
            pred = model(X) # foreground logit proba [N, 1]
            
            loss = nn.BCEWithLogitsLoss()(pred, y)
            y_true = y.detach().cpu().numpy()
            y_score = torch.sigmoid(pred).detach().cpu().numpy()

            losses = np.concatenate((losses, loss.detach().cpu().numpy().reshape(-1)))
            y_true_all = np.concatenate((y_true_all, y_true.reshape(-1)))
            y_score_all = np.concatenate((y_score_all, y_score.reshape(-1)))

            del X, y, pred, loss
            torch.cuda.empty_cache()
    
    y_pred_all = np.where(y_score_all > th, 1, 0).astype(np.uint8)
    accuracy = accuracy_score(y_true_all, y_pred_all)
    precision = precision_score(y_true_all, y_pred_all)
    recall = recall_score(y_true_all, y_pred_all)
    
    return np.average(losses), accuracy, precision, recall


def run_model(train_loader, val_loader, model, epochs, optim, scheduler, save_path): 
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    log_path = os.path.join(save_path, 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    
    #TODO(3) change to logging
    sys.stdout = utils.Logger(os.path.join(save_path, 'log'))
    tb_writer = SummaryWriter(log_path)
    
    #TODO(3)
    print('====================')
    print('start running at: ', utils.timestamp())
    start = utils.tic()
    
    min_loss = 65536
    for epoch in tqdm(range(1, epochs + 1), desc = 'Epoch'):
        print('---------------------') 
        torch.cuda.synchronize()
        epoch_start = utils.tic()
        print('start at: ', utils.timestamp())
        
        train_loss, train_acc, train_prc, train_rec = train(loader=train_loader, model=model, optim=optim)
        val_loss, val_acc, val_prc, val_rec = evaluate(loader=train_loader, model=model)
        scheduler.step()
        
        tb_writer.add_scalar('learning_rate', optim.param_groups[0]['lr'], global_step=epoch)
        tb_writer.add_scalars('Loss', {'train_loss': train_loss, 'val_loss': val_loss}, global_step=epoch)
        tb_writer.add_scalars('Accuracy', {'train_acc': train_acc, 'val_acc': val_acc}, global_step=epoch)
        tb_writer.add_scalars('Precision', {'train_precision': train_prc, 'val_precision': val_prc}, global_step=epoch)
        tb_writer.add_scalars('Recall', {'train_recall': train_rec, 'val_recall': val_rec}, global_step=epoch)
        
        torch.cuda.synchronize()
        epoch_end = utils.tic()
        print('end at: ', utils.timestamp())
        print('epoch runtime: ', utils.delta_time(epoch_start, epoch_end))
        
        #TODO(1): save model criteria
        if val_loss < min_loss:
            min_loss = val_loss
            ckpt_path = os.path.join(save_path, 'checkpoint.tar.gz')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optim.state_dict(),
                }, ckpt_path)
    
    print('---------------------')
    end = utils.tic()
    print('end running at: ', utils.timestamp())
    print('overall runtime: ', utils.delta_time(start, end))
    
    tb_writer.close()