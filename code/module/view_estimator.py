import torch
import torch.nn as nn
from .base import GCN_one, GCN_one_pyg
from torch_sparse import SparseTensor
import numpy as np

class GenView(nn.Module):
    def __init__(self, num_feature, hid, com_lambda, dropout, pyg):
        super(GenView, self).__init__()
        if pyg == False:
            self.gen_gcn = GCN_one(num_feature, hid, activation=nn.ReLU())
        else:
            self.gen_gcn = GCN_one_pyg(num_feature, hid, activation=nn.ReLU())  
        self.gen_mlp = nn.Linear(2 * hid, 1)
        nn.init.xavier_normal_(self.gen_mlp.weight, gain=1.414)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
        self.com_lambda = com_lambda
        self.dropout = nn.Dropout(dropout)

    def forward(self, v_ori, feat, v_indices, num_node):
        emb = self.gen_gcn(feat, v_ori)
        f1 = emb[v_indices[0]]
        f2 = emb[v_indices[1]]
        ff = torch.cat([f1, f2], dim=-1)
        temp = self.gen_mlp(self.dropout(ff)).reshape(-1)
        
        z_matrix = torch.sparse.FloatTensor(v_indices, temp, (num_node, num_node))
        pi = torch.sparse.softmax(z_matrix, dim=1)
        gen_v = v_ori + self.com_lambda * pi 
        return gen_v


class View_Estimator(nn.Module):
    def __init__(self, num_feature, gen_hid, com_lambda_v1, com_lambda_v2, dropout, pyg):
        super(View_Estimator, self).__init__()
        self.v1_gen = GenView(num_feature, gen_hid, com_lambda_v1, dropout, pyg)
        self.v2_gen = GenView(num_feature, gen_hid, com_lambda_v2, dropout, pyg)

    def forward(self, data):
        new_v1 = data.normalize(self.v1_gen(data.view1, data.x, data.v1_indices, data.num_node))
        new_v2 = data.normalize(self.v2_gen(data.view2, data.x, data.v2_indices, data.num_node))
        return new_v1, new_v2
