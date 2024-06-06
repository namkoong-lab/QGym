
from torch import nn
import torch.nn.functional as F
import torch

class PriorityNet(nn.Module):
    def __init__(self, s, q, layers, hidden_dim, f_time = False, x_stats = None, t_stats = None):
        super().__init__()
        self.s = s
        self.q = q
        self.x_stats = x_stats
        self.t_stats = t_stats
        self.layers = layers
        self.hidden_dim = hidden_dim
        
        self.f_time = f_time
        
        if self.f_time:
            self.input_fc = nn.Linear(self.q + 1, hidden_dim)
        else:
            self.input_fc = nn.Linear(self.q, hidden_dim)
            
        self.layers_fc = nn.ModuleList()
        for _ in range(layers):
            self.layers_fc.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.output_fc = nn.Linear(hidden_dim, self.s * self.q)
        #self.output_fc = nn.Linear(hidden_dim, self.q * self.s)
        
    def forward(self, x, t = 0):
        
        # Input layer
        batch = x.size()[0]
        
        if self.x_stats is not None:
            x = (x - self.x_stats[0]) / self.x_stats[1]

        if self.t_stats is not None:    
            t = (t - self.t_stats[0]) / self.t_stats[1]
        
        if self.f_time:
            x = torch.cat((x, t), 1)
            
        x = F.relu(self.input_fc(x))

        x = F.relu(self.layers_fc[0](x))

        # Hidden layer
        for l in range(self.layers):
            x = F.relu(self.layers_fc[l](x))

        # Output layer
        x = self.output_fc(x)
        return F.softmax(torch.reshape(x, (batch, self.s , self.q)), dim = 2)

class DirectBackpropPolicy:
    def __init__(self, network):
        self.network = network
        
    def test_forward(self, step, batch_queue, batch_time, repeated_queue, repeated_network, repeated_mu, repeated_h):
        return self.network(batch_queue, batch_time)
    
    def train_forward(self, queues, time, network, repeated_h, mu):
        return self.network(queues, time)

