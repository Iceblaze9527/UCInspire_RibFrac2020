import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from tqdm import tqdm

import utils

def train(loader, model, optim):
    model.train()
    losses = []
    acces = []
    ##TODO(1) precision and recall and others
    
    for idx, (X, y) in tqdm(enumerate(loader), total=len(loader), desc='Training'):
        optim.zero_grad()
        X = X.float().cuda() # [N, IC=1, D, H, W]
        y = y.long().cuda() # foreground=1
        pred = model(X) # foreground logit proba
        
        loss = F.cross_entropy(pred, y)
        acc = 1 if torch.sigmoid(pred).detach().cpu().numpy() > 0.5 else 0 ##TODO(1) output dimensiom
        losses.append(loss.detach().cpu().numpy())
        acces.append(acc)
        
        loss.backward()
        optim.step()

        del X, y, pred, loss
        torch.cuda.empty_cache()
    
    return np.average(losses), np.average(acces)

def evaluate(loader, model):
    model.eval()
    losses = []
    acces = []
    ##TODO(1) precision and recall and others
    
    with torch.no_grad():
        for idx, (X, y) in tqdm(enumerate(loader), total=len(loader), desc='Validating'):
            X = X.float().cuda() # [N, IC=1, D, H, W]
            y = y.float().cuda() # foreground=1
            pred = model(X) # foreground logit proba
            
            loss = F.cross_entropy(pred, y)
            acc = 1 if torch.sigmoid(pred).detach().cpu().numpy() > 0.5 else 0 ##TODO(1) output dimensiom
            losses.append(loss.detach().cpu().numpy())
            acces.append(acc)

            del X, y, pred, loss
            torch.cuda.empty_cache()
    
    return np.average(losses), np.average(acces)


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
    
    for epoch in tqdm(range(1, epochs + 1), desc = 'Epoch'):
        print('---------------------') 
        torch.cuda.synchronize()
        epoch_start = utils.tic()
        print('start at: ', utils.timestamp())
        
        train_loss, train_acc = train(loader=train_loader, model=model, optim=optim)
        val_loss, val_acc = evaluate(loader=train_loader, model=model)
        scheduler.step()
        
        lr = optim.param_groups[0]['lr']
        tb_writer.add_scalar('learning_rate', global_step=epoch)
        tb_writer.add_scalars('Loss', {'train_loss': train_loss, 'val_loss': val_loss}, global_step=epoch)
        tb_writer.add_scalars('Accuracy', {'train_acc': train_acc, 'val_acc': val_acc}, global_step=epoch)
        
        torch.cuda.synchronize()
        epoch_end = utils.tic()
        print('end at: ', utils.timestamp())
        print('epoch runtime: ', utils.delta_time(epoch_start, epoch_end))
        
        #TODO(1): save model criteria
        if val_loss < min(all_val_losses + [256]):
            ckpt_path = os.path.join(save_path, 'checkpoint.tar.gz')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optim_state_dict': optim.state_dict(),
                }, ckpt_path)
    
    print('---------------------')
    end = utils.tic()
    print('end running at: ', utils.timestamp())
    print('overall runtime: ', utils.delta_time(start, end))
    
    tb_writer.close()