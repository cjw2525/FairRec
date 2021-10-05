from utils.metrics import metrics
import numpy as np
import torch
import math
from tqdm import tqdm
import torch.nn as nn
        
def mklogs():
    logs = {'train_loss':[], 
            'train_f_loss':[], 
            'RMSE':[],
            'acc':[], 
            'DEE':[], 
            'DP_user':[], 
            'DP_item':[], 
            'v_fairness':[]
           }
    return logs 

def get_test_logs(logs, log):
    measures=['RMSE',
              'acc', 
              'DEE', 
              'DP_user', 
              'DP_item', 
              'v_fairness']
    for measure in measures:
        logs[measure].append(log[measure])
        
def train_PQ(data_tuple, model, optimizer, 
             num_epochs, device, l_value=0., lambda_=0., f_criterion=None, tau=3):

    logs = mklogs()
    data, gender, item = data_tuple
    
    # data_input 
    train_data = torch.from_numpy(data[0]).float().to(device) 
    test_data = data[1]
    
    identity = torch.from_numpy(np.eye(data[0].shape[0])).float().to(device)
    
    x = train_data
    mask = x.clone().detach()
    mask = torch.where(mask != 0, torch.ones(1).to(device), torch.zeros(1).to(device)).float().to(device)
    count = torch.sum(mask).item()
    
    #losses
    criterion = nn.MSELoss(reduction='sum')
    
    for epoch in range(num_epochs):
        rmse, cost = 0, 0
        model.train()
        W, V = model.encoder[0].weight, model.decoder[0].weight
        W_fro, V_fro = torch.sum(W ** 2), torch.sum(V ** 2)
        
        x_hat = model(identity)
        loss = 0 
        loss += (1-lambda_)*(criterion(x * mask, x_hat * mask)/count + l_value / 2 * ( W_fro + V_fro ))
        if f_criterion!=None: 
            f_loss = f_criterion(x_hat, gender, item)
            logs['train_f_loss'].append(f_loss.item())
            loss += lambda_*f_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logs['train_loss'].append(loss.item())
        
    return logs

def train_AE(data_tuple, model, optimizer, 
             num_epochs, device, l_value=0., lambda_=0., f_criterion=None, tau=3):

    logs = mklogs()
    data, gender, item = data_tuple
    
    # data_input 
    train_data = torch.from_numpy(data[0]).float().to(device) 
    test_data = data[1]
    
    x = train_data
    mask = x.clone().detach()
    mask = torch.where(mask != 0, torch.ones(1).to(device), torch.zeros(1).to(device)).float().to(device)
    count = torch.sum(mask).item()
    
    #losses
    criterion = nn.MSELoss(reduction='sum')
    
    for epoch in range(num_epochs):
        rmse, cost = 0, 0
        model.train()
        W, V = model.encoder[0].weight, model.decoder[0].weight
        W_fro, V_fro = torch.sum(W ** 2), torch.sum(V ** 2)
        
        x_hat = model(x)
        loss = 0 
        loss += (1-lambda_)*(criterion(x * mask, x_hat * mask)/count + l_value / 2 * ( W_fro + V_fro ))
        if f_criterion!=None: 
            f_loss = f_criterion(x_hat, gender, item)
            logs['train_f_loss'].append(f_loss.item())
            loss += lambda_*f_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        logs['train_loss'].append(loss.item())
        
    return logs