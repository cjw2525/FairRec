import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, rating, num_user):
        super(AE, self).__init__()
        
        self.rating = rating
        self.encoder = nn.Sequential(
          nn.Linear(num_user, 512),
          nn.ReLU(),
          nn.Linear(512, 512),
          nn.Dropout(0.7),
          nn.ReLU(),
        )
        self.decoder = nn.Sequential(
          nn.Linear(512, num_user),
        )
        
    def forward(self, x):
        x = torch.transpose(x,0,1)
        if self.rating == 'binary':
            x = self.decoder(self.encoder(x))
            x = torch.tanh(x)
        elif self.rating == 'five-stars':
            x = (x - 1) / 4.0
            x = self.decoder(self.encoder(x))
            x = torch.clamp(x, 0, 1.0)
            x = 4.0 * x + 1
        x = torch.transpose(x,0,1)
        return x

