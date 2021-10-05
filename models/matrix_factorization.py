import torch
import torch.nn as nn
    
class PQ(nn.Module):
    def __init__(self, rating, num_users, num_items, rank):
        super(PQ, self).__init__()
        
        self.rating = rating
        
        self.encoder = nn.Sequential(nn.Linear(num_users, rank, bias=False))
        self.decoder = nn.Sequential(nn.Linear(rank, num_items, bias=False))

    def forward(self, x):
        if self.rating == 'binary':
            x = self.decoder(self.encoder(x))
            x = torch.tanh(x)
        elif self.rating == 'five-stars':
            x = self.decoder(self.encoder(x))
            x = torch.clamp(x, 0, 5.0)
        else:
            raise KeyError("unavailable rating scale")
    
        return x