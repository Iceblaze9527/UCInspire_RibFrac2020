import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np

import sys
import time
import datetime
import timeit

from evaluation import binary_iou
import losses

def train_model(device, loader, model, optim, loss_func, gamma, alpha, pad):
    model.train()
    train_loss = 0
    train_jaccard = 0
    
    for idx, (X, y) in enumerate(loader):
        optim.zero_grad()
        X = X.to(device, dtype=torch.float)# [N, IC, H, W]
        y = y.to(device, dtype=torch.long)# [N, H, W] with class indices (0, 1)
        
        pred = model(X)# [N, OC, H, W]
        if pad > 0:
            pred = pred[:, :, pad:-pad, pad:-pad]#crop predictions if there's padding
        
        if loss_func == 'focal_loss':
            loss = losses.FocalLoss(gamma=gamma, alpha=alpha)(pred, y)#mean loss
        elif loss_func == 'cross_entropy':
            loss = F.cross_entropy(pred, y, reduction='mean')#mean loss
        else:
            raise ValueError('No such loss function. Only focal_loss and cross_entropy are available.')
        train_loss += (loss.detach().cpu().numpy()) * y.shape[0]#sum of loss in this batch
        
        pred_act = torch.sigmoid(pred)[:,0,:,:]#sigmoid activation, size[N, 1, H, W]
        if (torch.median(pred_act) > torch.mean(pred_act)):#choose the channel whose background is zero
            pred_act = 1 - pred_act
        pred_eval = pred_act.detach().cpu().numpy()
        y_eval = y.detach().cpu().numpy()
        train_jaccard += binary_iou(pred_eval, y_eval)#sum of jaccard in this batch
        
        loss.backward()
        optim.step()

        del X, y, pred, pred_act, loss
        torch.cuda.empty_cache()
    
    return train_loss, train_jaccard

def val_model(device, loader, model, loss_func, gamma, alpha, pad):
    model.eval()
    val_loss = 0
    val_jaccard = 0
    
    with torch.no_grad():
        for idx, (X, y) in enumerate(loader):
            X = X.to(device, dtype=torch.float)# [N, IC, H, W] 
            y = y.to(device, dtype=torch.long)# [N, H, W] with class indices (0, 1)

            pred = model(X)# [N, OC, H, W]
            if pad > 0:
                pred = pred[:, :, pad:-pad, pad:-pad]
            
            if loss_func == 'focal_loss':
                loss = losses.FocalLoss(gamma=gamma, alpha=alpha)(pred, y)
            elif loss_func == 'cross_entropy':
                loss = F.cross_entropy(pred, y, reduction='mean')
            else:
                raise ValueError('No such loss function. Only focal_loss and cross_entropy are available.')
            val_loss += (loss.detach().cpu().numpy()) * y.shape[0]
            
            pred_act = torch.sigmoid(pred)[:,0,:,:]
            if (torch.median(pred_act) > torch.mean(pred_act)):
                pred_act = 1 - pred_act
            pred_eval = pred_act.detach().cpu().numpy()
            y_eval = y.detach().cpu().numpy()
            val_jaccard += binary_iou(pred_eval, y_eval)

            del X, y, pred, pred_act, loss
            torch.cuda.empty_cache()
    
    return val_loss, val_jaccard

def run_model(device, train_dataset, val_dataset, model, optim, loss_func, gamma, alpha, epochs, pad, batch_size, save_path): 
    all_train_losses = []
    all_train_jaccards = []
    all_val_losses = []
    all_val_jaccards = []
    
    train_size = len(train_dataset)
    val_size = len(val_dataset)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    print('start running at: ', time.asctime(time.localtime(time.time())))
    start = timeit.default_timer()
    
    for epoch in range(epochs):
        print('epoch:',epoch+1)
        torch.cuda.synchronize()
        epoch_start = timeit.default_timer()
        print('start at: ', time.asctime(time.localtime(time.time())))
        
        train_loss, train_jaccard = train_model(device=device, 
                                                loader=train_loader, 
                                                model=model,
                                                optim=optim, 
                                                loss_func=loss_func,  
                                                gamma=gamma, 
                                                alpha=alpha,
                                                pad=pad)
        
        val_loss, val_jaccard = val_model(device=device, 
                                        loader=val_loader,
                                        model=model,
                                        loss_func=loss_func,  
                                        gamma=gamma, 
                                        alpha=alpha,
                                        pad=pad)
        
        torch.cuda.synchronize()
        epoch_end = timeit.default_timer()
        print('runtime:', str(datetime.timedelta(seconds=round(epoch_end-epoch_start,3))))
        
        train_loss /= train_size
        train_jaccard /= train_size
        val_loss /= val_size
        val_jaccard /= val_size
        
        print('train_loss:', train_loss)
        print('train_jaccard:', train_jaccard)
        print('val_loss:', val_loss)
        print('val_jaccard:', val_jaccard)
        
        #save best results
        if val_loss < min(all_val_losses + [256]):
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'loss': train_loss,
                }, save_path)
            
        all_train_losses.append(train_loss)
        all_train_jaccards.append(train_jaccard)
        all_val_losses.append(val_loss)
        all_val_jaccards.append(val_jaccard)
    
    print('end running at: ', time.asctime(time.localtime(time.time())))
    end = timeit.default_timer()
    print('overall runtime:', str(datetime.timedelta(seconds=round(end-start,3))))
    
    return (all_train_losses, all_train_jaccards, all_val_losses, all_val_jaccards)