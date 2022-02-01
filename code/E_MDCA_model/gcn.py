import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as sp
from scipy.special import softmax
import torch.nn.functional as Func


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, acti=True):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim) # bias = False is also ok.
        if acti:
            self.acti = nn.ReLU(inplace=True)
        else:
            self.acti = None
    def forward(self, F):
        output = self.linear(F)
        if not self.acti:
            return output
        return self.acti(output)


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, device, comp_size, p=0.0):
        super(GCN, self).__init__()
        self.device = device
#         self.norm1 = nn.BatchNorm1d(input_dim)
        self.gcn_layer1 = GCNLayer(input_dim, hidden_dim)
#         self.gcn_layer2 = GCNLayer(hidden_dim, hidden_dim, acti=False)
#         self.dropout = nn.Dropout(p)
#         self.norm2 = nn.LayerNorm(hidden_dim)
#         self.gcn_layer1 = nn.Linear(comp_size * hidden_dim, hidden_dim)
        self.comp_size = comp_size
        self.hidden_dim = hidden_dim
        self.flatten = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, A, X, comp_idx):
        
#         X = self.dropout(X)
#         print(X.size(), A.size())
        F = torch.matmul(A, X)
        F = self.gcn_layer1(F)
        output = Func.relu(F)
#         F = self.dropout(F)
#         print(F.size())
#         F = self.norm2(F)
#         F = torch.matmul(A, F)
#         output = self.gcn_layer2(F)
#         print(output.size())
        output = output[[comp_idx]]
        output = self.flatten(output)
#         print(output.size())
        return output