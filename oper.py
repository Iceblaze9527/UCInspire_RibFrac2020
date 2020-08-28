import os

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm

import utils
from metrics import metrics


concat = lambda head, tail: np.concatenate((head, tail), axis=0) if head.size else tail

def train(loader, model, optim, criterion):
    model.train()
    
    losses = np.array([])
    y_name_all = []
    y_center_all = np.array([])
    y_true_all = np.array([])
    y_score_all = np.array([])
    
    for idx, (X, y_src) in tqdm(enumerate(loader), total=len(loader), desc='Training'):
        optim.zero_grad()
        
        y, y_name, y_center = y_src
        X = X.float().cuda() # [N, IC=1, D, H, W]
        y = y.float().cuda() # [N]
        pred = model(X) # logit proba [N, OC]

        loss = criterion(pred, y)
        y_true = y.detach().cpu().numpy().astype(np.uint8)
        
        if pred.size()[1] == 1:
            y_score = torch.sigmoid(pred).detach().cpu().numpy()
        else:
            y_score = F.softmax(pred, dim=1).detach().cpu().numpy()
        
        losses = concat(losses, loss.detach().cpu().numpy())
        y_name_all.extend(y_name)
        y_center_all = concat(y_center_all, y_center)
        y_true_all = concat(y_true_all, y_true)
        y_score_all = concat(y_score_all, y_score)
        
        loss.mean().backward()
        optim.step()

        del X, y, pred, loss
        torch.cuda.empty_cache()
    
    return losses, y_name_all, y_center_all, y_true_all, y_score_all


def evaluate(loader, model, criterion):
    model.eval()
    
    losses = np.array([])
    y_name_all = []
    y_center_all = np.array([])
    y_true_all = np.array([])
    y_score_all = np.array([])
    
    with torch.no_grad():
        for idx, (X, y_src) in tqdm(enumerate(loader), total=len(loader), desc='Validating'):
            y, y_name, y_center = y_src
            X = X.float().cuda() # [N, IC=1, D, H, W]
            y = y.float().cuda() # [N]
            pred = model(X) # logit proba [N, OC]
            
            loss = criterion(pred, y)
            y_true = y.detach().cpu().numpy().astype(np.uint8)
            
            if pred.size()[1] == 1:
                y_score = torch.sigmoid(pred).detach().cpu().numpy()
            else:
                y_score = F.softmax(pred, dim=1).detach().cpu().numpy()
            
            losses = concat(losses, loss.detach().cpu().numpy())
            y_name_all.extend(y_name)
            y_center_all = concat(y_center_all, y_center)
            y_true_all = concat(y_true_all, y_true)
            y_score_all = concat(y_score_all, y_score)

            del X, y, pred, loss
            torch.cuda.empty_cache()
    
    return losses, y_name_all, y_center_all, y_true_all, y_score_all


def test(loader, model):
    model.eval()
    
    y_name_all = []
    y_center_all = np.array([])
    y_score_all = np.array([])
    
    with torch.no_grad():
        for idx, (X, y_src) in tqdm(enumerate(loader), total=len(loader), desc='Testing'):
            X = X.float().cuda() # [N, IC=1, D, H, W]
            pred = model(X) # logit proba [N, OC]
            
            y_name, y_center = y_src
            y_score = torch.sigmoid(pred).detach().cpu().numpy()

            y_name_all.extend(y_name)
            y_center_all = concat(y_center_all, y_center)
            y_score_all = concat(y_score_all, y_score)

            del X, pred
            torch.cuda.empty_cache()
    
    return y_name_all, y_center_all, y_score_all


def run(train_loader, val_loader, model, is_multi, epochs, optim, criterion, scheduler, save_path):
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
        
        train_results = train(loader=train_loader, model=model, optim=optim, criterion=criterion)
        val_results = evaluate(loader=val_loader, model=model, criterion=criterion)
        
        scheduler.step()
        
        torch.cuda.synchronize()
        print('end at: ', utils.timestamp())
        epoch_end = utils.tic()
        print('epoch runtime: ', utils.delta_time(epoch_start, epoch_end))
        
        train_loss, train_acc, train_prc, train_rec, train_roc_auc = metrics(
            train_results, csv_path = os.path.join(data_path, 'train_%02d.csv'%(epoch)), is_multi=is_multi, is_test=False)
       
        val_loss, val_acc, val_prc, val_rec, val_roc_auc = metrics(
            val_results, csv_path = os.path.join(data_path, 'val_%02d.csv'%(epoch)), is_multi=is_multi, is_test=False)
        
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
        
        for name, value in model.named_parameters():
            tb_writer.add_histogram(name, value.data.cpu().numpy(), global_step=epoch)
            tb_writer.add_histogram(name + '/grad', value.grad.data.cpu().numpy(), global_step=epoch)
        
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