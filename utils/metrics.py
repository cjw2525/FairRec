import torch
import numpy as np

def metrics(model, data_tuple, device, model_type='AE', tau=3):
    
    data, gender, item = data_tuple
    measures = {}
    
    with torch.no_grad(): 
        model.eval()
        Y_train, Y_test = data[0], data[1]
        if model_type=='PQ':
            identity = torch.from_numpy(np.eye(Y_train.shape[0])).float().to(device)
            pred = model(identity).cpu().detach().numpy()
        else:
            pred = model(torch.tensor(Y_train).float().to(device)).cpu().detach().numpy()   
        # 1. rmse
        test_rmse = np.sqrt(np.mean((Y_test[Y_test != 0] - pred[Y_test != 0]) ** 2))
        # Y_tilde 
        pred_hat = np.where(pred > tau, 1, 0)
        
        # 2. DEE
        DEE = 0
        for g in ['M', 'F']:
            for i in ['M', 'F']:
                DEE += np.abs(np.mean(pred_hat)-np.mean(pred_hat[gender[g]][:, item[i]]))
        # 3. value_fairness
        VAL = VAL_measure(pred, data, gender, device)
        # 4. DP_user
        UGF = 0
        for g in ['M', 'F']:
            UGF += np.abs(np.mean(pred_hat)-np.mean(pred_hat[gender[g]]))
        # 4. DP_item
        CVS = 0
        for i in ['M', 'F']:
            CVS += np.abs(np.mean(pred_hat)-np.mean(pred_hat[:, item[i]]))
        measures['RMSE'] = test_rmse
        measures['DEE'] = DEE
        measures['VAL'] = VAL 
        measures['UGF'] = UGF
        measures['CVS'] = CVS
    return measures

def VAL_measure(pred, data, gender, device):
    train_data = data[0]
    mask = np.where(train_data!=0, 1, 0)

    y_m = train_data[gender['M']]
    y_f = train_data[gender['F']]
    y_hat_m = pred[gender['M']]
    y_hat_f = pred[gender['F']]

    #average ratings
    d_m = np.abs(np.sum(y_m, axis=0)/(np.sum(mask[gender['M']], axis=0)+1e-8)-np.sum(y_hat_m, axis=0)/len(gender['M']))
    d_f = np.abs(np.sum(y_f, axis=0)/(np.sum(mask[gender['F']], axis=0)+1e-8)-np.sum(y_hat_f, axis=0)/len(gender['F']))

    v_fairness = np.mean(np.abs(d_m-d_f))
    return v_fairness
        