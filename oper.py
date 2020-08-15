import sys
import time
import datetime
import timeit

import numpy as np
import torch
import torch.nn.functional as F

def train(loader, model, optim):
    model.train()
    
    for idx, (X, y) in enumerate(loader):
        optim.zero_grad()
        X = X.float().cuda() # [N, IC=1, D, H, W]
        y = y.long().cuda()# foreground=1
        pred = model(X)# foreground proba
        
        loss = F.cross_entropy(pred, y)
        
        loss.backward()
        optim.step()

        del X, y, pred, loss
        torch.cuda.empty_cache()

def evaluate(loader, model):
    model.eval()
    
    with torch.no_grad():
        for idx, (X, y) in enumerate(loader):
            X = X.float().cuda()# [N, IC=1, D, H, W]
            y = y.float().cuda()# foreground=1
            pred = model(X)# foreground proba
            
            loss = F.cross_entropy(pred, y)

            del X, y, pred, loss
            torch.cuda.empty_cache()

def run_model(train_loader, val_loader, model, optim, epochs, batch_size, save_path): 
    #TODO(1): running logs
    print('start running at: ', time.asctime(time.localtime(time.time())))
    start = timeit.default_timer()
    
    for epoch in range(epochs):
        #TODO(1): running logs
        print('\nepoch:',epoch+1)
        torch.cuda.synchronize()
        epoch_start = timeit.default_timer()
        print('start at: ', time.asctime(time.localtime(time.time())))
        
        train(loader=train_loader, model=model, optim=optim)
        evaluate(loader=train_loader, model=model)
        
        #TODO(1): running logs
        torch.cuda.synchronize()
        epoch_end = timeit.default_timer()
        print('runtime: ', str(datetime.timedelta(seconds=round(epoch_end-epoch_start,3))))
        
        #TODO(2): save model criteria
        if val_loss < min(all_val_losses + [256]):
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optim.state_dict(),
                'loss': train_loss,
                }, save_path)
            
    #TODO(1): running logs
    print('end running at: ', time.asctime(time.localtime(time.time())))
    end = timeit.default_timer()
    print('overall runtime: ', str(datetime.timedelta(seconds=round(end-start,3))))