import os

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm

import utils
from metrics import metrics

def train(loader, model, optim):
    model.train()
    
    losses = np.array([])
    y_true_all = np.array([])
    y_score_all = np.array([])
    
    for idx, (X, y) in tqdm(enumerate(loader), total=len(loader), desc='Training'):
        optim.zero_grad()
        X = X.float().cuda() # [N, IC=1, D, H, W]
        y = y.float().cuda() # foreground = 1
        pred = model(X) # foreground logit proba [N, 1]
        
        loss = nn.BCEWithLogitsLoss()(pred, y)
        y_true = y.detach().cpu().numpy().astype(np.uint8)
        y_score = torch.sigmoid(pred).detach().cpu().numpy()
        
        losses = np.concatenate((losses, loss.detach().cpu().numpy().reshape(-1)))
        y_true_all = np.concatenate((y_true_all, y_true.reshape(-1)))
        y_score_all = np.concatenate((y_score_all, y_score.reshape(-1)))
        
        loss.backward()
        optim.step()

        del X, y, pred, loss
        torch.cuda.empty_cache()
    
    return losses, y_true_all, y_score_all


def evaluate(loader, model):
    model.eval()

    losses = np.array([])
    y_true_all = np.array([])
    y_score_all = np.array([])
    
    with torch.no_grad():
        for idx, (X, y) in tqdm(enumerate(loader), total=len(loader), desc='Validating'):
            X = X.float().cuda() # [N, IC=1, D, H, W]
            y = y.float().cuda() # foreground = 1
            pred = model(X) # foreground logit proba [N, 1]
            
            loss = nn.BCEWithLogitsLoss()(pred, y)
            y_true = y.detach().cpu().numpy().astype(np.uint8)
            y_score = torch.sigmoid(pred).detach().cpu().numpy()

            losses = np.concatenate((losses, loss.detach().cpu().numpy().reshape(-1)))
            y_true_all = np.concatenate((y_true_all, y_true.reshape(-1)))
            y_score_all = np.concatenate((y_score_all, y_score.reshape(-1)))

            del X, y, pred, loss
            torch.cuda.empty_cache()
    
    return losses, y_true_all, y_score_all


def run(train_loader, val_loader, model, epochs, optim, scheduler, save_path, threshold):
    ckpt_path = os.path.join(save_path, 'checkpoint.tar.gz')
    log_path = os.path.join(save_path, 'logs')
    data_path = os.path.join(save_path, 'data')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        
    tb_writer = SummaryWriter(log_path)
    
    #TODO(3): logger module
    print('====================')
    print('start running at: ', utils.timestamp())
    start = utils.tic()
    
    #TODO(3): save criteria
    min_loss = 65536
    print('====================')
    for epoch in tqdm(range(1, epochs + 1), desc = 'Epoch'): 
        torch.cuda.synchronize()
        print(f'Epoch {epoch}:')
        print('start at: ', utils.timestamp())
        epoch_start = utils.tic()
        
        train_losses, train_y_true, train_y_score = train(loader=train_loader, model=model, optim=optim)
        val_losses, val_y_true, val_y_score = evaluate(loader=val_loader, model=model)
        
        scheduler.step()
        
        torch.cuda.synchronize()
        print('end at: ', utils.timestamp())
        epoch_end = utils.tic()
        print('epoch runtime: ', utils.delta_time(epoch_start, epoch_end))
       
        train_loss = np.average(train_losses)
        train_acc, train_prc, train_rec, train_roc_auc, train_curve = metrics(
            train_y_true, train_y_score, threshold=threshold, csv_path = os.path.join(data_path, 'train_%02d.csv'%(epoch)))
        
        val_loss = np.average(val_losses) 
        val_acc, val_prc, val_rec, val_roc_auc, val_curve = metrics(
            val_y_true, val_y_score, threshold=threshold, csv_path = os.path.join(data_path, 'val_%02d.csv'%(epoch)))
        
        print('---------------------')
        print('Train Results:')
        print('Loss: ', train_loss)
        print('Accuracy: ', train_acc)
        print('Precision:', train_prc)
        print('Recall:', train_rec)
        print('ROC AUC:', train_roc_auc)
        
        print('---------------------')
        print('Validation Results:')
        print('Loss: ', val_loss)
        print('Accuracy: ', val_acc)
        print('Precision:', val_prc)
        print('Recall:', val_rec)
        print('ROC AUC:', val_roc_auc)
        
        #TODO(3) tensorboard logger
        tb_writer.add_scalar('learning_rate', optim.param_groups[0]['lr'], global_step=epoch)
        tb_writer.add_scalars('Loss', {'train_loss': train_loss, 'val_loss': val_loss}, global_step=epoch)
        tb_writer.add_scalars('Accuracy', {'train_acc': train_acc, 'val_acc': val_acc}, global_step=epoch)
        tb_writer.add_scalars('Precision', {'train_precision': train_prc, 'val_precision': val_prc}, global_step=epoch)
        tb_writer.add_scalars('Recall', {'train_recall': train_rec, 'val_recall': val_rec}, global_step=epoch)
        tb_writer.add_scalars('ROC AUC', {'train_roc_auc': train_roc_auc, 'val_roc_auc': val_roc_auc}, global_step=epoch)
        tb_writer.add_figure('ROC curves', [train_curve, val_curve], global_step=epoch)##
        
        for name, value in model.named_parameters():
            tb_writer.add_histogram(name, value.data.cpu().numpy(), global_step=epoch)
            tb_writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(),global_step=epoch)
        
        #TODO(3): save criteria
        if val_loss < min_loss:
            min_loss = val_loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 
                        'optim_state_dict': optim.state_dict(),}, ckpt_path)
        
        print('====================')
    
    print('end running at: ', utils.timestamp())
    end = utils.tic()
    print('overall runtime: ', utils.delta_time(start, end))
    print('====================')
    tb_writer.close()