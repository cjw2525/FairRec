import numpy as np
import torch
import math, collections, itertools, random

def normal_pdf(x):
    import math
    return torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

def normal_cdf(y, h=0.01, tau=0.5):
    # Approximation of Q-function given by López-Benítez & Casadevall (2011)
    # based on a second-order exponential function & Q(x) = 1 - Q(-x):
    Q_fn = lambda x: torch.exp(-0.4920*x**2 - 0.2887*x - 1.1893)
    m = y.shape[0]*y.shape[1]
    y_prime = (tau - y) / h
    sum_ = torch.sum(Q_fn(y_prime[y_prime > 0])) \
           + torch.sum(1 - Q_fn(torch.abs(y_prime[y_prime < 0]))) \
           + 0.5 * len(y_prime[y_prime == 0])
    return sum_ / m

def Huber_loss(x, delta):
    if abs(x) < delta:
        return (x ** 2) / 2
    return delta * (x.abs() - delta / 2)

def Huber_loss_derivative(x, delta):
    if x > delta:
        return delta/2
    elif x < -delta:
        return -delta/2
    return x

class FairnessLoss():
    def __init__(self, h, tau, delta, device, data_tuple, type_='ours'):
        self.h = h
        self.tau = tau
        self.delta = delta
        self.device = device
        self.type_ = type_
        self.data_tuple = data_tuple

    def DEE(self, y_hat, gender, item):
        backward_loss = 0
        logging_loss_ = 0 
        
        for gender_key in ['M','F']:
            for item_key in ['M', 'F']:
                gender_idx = gender[gender_key] 
                item_idx = item[item_key]
                m_gi = len(gender_idx)*len(item_idx)
                y_hat_gender_item = y_hat[gender_idx][:, item_idx]

                Prob_diff_Z = normal_cdf(y_hat.detach(), self.h, self.tau)-normal_cdf(y_hat_gender_item.detach(), self.h, self.tau)
                
                _dummy = Huber_loss_derivative(Prob_diff_Z, self.delta)
                _dummy *= \
                    torch.dot(
                        normal_pdf((self.tau - y_hat.detach()) / self.h).reshape(-1), 
                        y_hat.reshape(-1)
                    ) / (self.h * y_hat.shape[0]*y_hat.shape[1]) -\
                    torch.dot(
                        normal_pdf((self.tau - y_hat_gender_item.detach()) / self.h).reshape(-1), 
                        y_hat_gender_item.reshape(-1)
                    ) / (self.h * m_gi)
                backward_loss += _dummy
        return backward_loss
        
    def VAL(self, y_hat, gender, item):
        device = self.device
        
        backward_loss = 0
        
        data = self.data_tuple[0]
        train_data = data[0]
        mask = np.where(train_data!=0, 1, 0)

        train_data = torch.from_numpy(train_data).to(device)
        mask = torch.from_numpy(mask).to(device)

        y_m = train_data[gender['M']]
        y_f = train_data[gender['F']]
        y_hat_m = y_hat[gender['M']]
        y_hat_f = y_hat[gender['F']]

        #average ratings
        d_m = torch.abs(torch.sum(y_m, axis=0)/(torch.sum(mask[gender['M']], axis=0)+1e-8)
        -torch.sum(y_hat_m, axis=0)/len(gender['M']))

        d_f = torch.abs(torch.sum(y_f, axis=0)/(torch.sum(mask[gender['F']], axis=0)+1e-8)
        -torch.sum(y_hat_f, axis=0)/len(gender['F']))


        backward_loss = torch.mean(torch.abs(d_m-d_f))
        
        return backward_loss
    
    def UGF(self, y_hat, gender, item):
        backward_loss = 0
        
        for key in ['M', 'F']:
            
            gender_idx = gender[key]
            m_i = y_hat.shape[1]*len(gender_idx)
            y_hat_group = y_hat[gender_idx]
            
            Prob_diff_Z = normal_cdf(y_hat.detach(), self.h, self.tau)-normal_cdf(y_hat_group.detach(), self.h, self.tau)

            _dummy = Huber_loss_derivative(Prob_diff_Z, self.delta)
            _dummy *= \
                torch.dot(
                    normal_pdf((self.tau - y_hat.detach()) / self.h).reshape(-1), 
                    y_hat.reshape(-1)
                ) / (self.h * y_hat.shape[0]*y_hat.shape[1]) -\
                torch.dot(
                    normal_pdf((self.tau - y_hat_group.detach()) / self.h).reshape(-1), 
                    y_hat_group.reshape(-1)
                ) / (self.h * m_i)
            backward_loss += _dummy
        return backward_loss
    
    def CVS(self, y_hat, gender, item):
        backward_loss = 0
        
        for key in ['M', 'F']:
            item_idx = item[key]
            m_i = y_hat.shape[0]*len(item_idx)
            y_hat_group = y_hat[:, item_idx]

            Prob_diff_Z = normal_cdf(y_hat.detach(), self.h, self.tau)-normal_cdf(y_hat_group.detach(), self.h, self.tau)

            _dummy = Huber_loss_derivative(Prob_diff_Z, self.delta)
            _dummy *= \
                torch.dot(
                    normal_pdf((self.tau - y_hat.detach()) / self.h).reshape(-1), 
                    y_hat.reshape(-1)
                ) / (self.h * y_hat.shape[0]*y_hat.shape[1]) -\
                torch.dot(
                    normal_pdf((self.tau - y_hat_group.detach()) / self.h).reshape(-1), 
                    y_hat_group.reshape(-1)
                ) / (self.h * m_i)
            backward_loss += _dummy
        return backward_loss
        
    
    def __call__(self, y_hat, gender, item):
        if self.type_ == 'ours':
            return self.DEE(y_hat, gender, item)
        elif self.type_ == 'VAL':
            return self.VAL(y_hat, gender, item)
        elif self.type_ == 'UGF':
            return self.UGF(y_hat, gender, item)
        elif self.type_ == 'CVS':
            return self.CVS(y_hat, gender, item)
        