import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor

class GCN_two_pyg(nn.Module):
    def __init__(self, input_dim, hid_dim1, hid_dim2, dropout=0., activation="relu"):
        super(GCN_two_pyg, self).__init__()
        self.conv1 = GCNConv(input_dim, hid_dim1)
        self.conv2 = GCNConv(hid_dim1, hid_dim2)

        self.dropout = dropout
        assert activation in ["relu", "leaky_relu", "elu"]
        self.activation = getattr(F, activation)

    def forward(self, feature, adj):
        adj = SparseTensor.from_dense(adj.to_dense())
        x1 = self.activation(self.conv1(feature, adj))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.conv2(x1, adj)
        return x2

class GCN_one_pyg(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True, activation=None):
        super(GCN_one_pyg, self).__init__()
        self.conv1 = GCNConv(in_ft, out_ft)
        self.activation = activation
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

    def forward(self, feat, adj):
        adj = SparseTensor.from_dense(adj.to_dense())
        out = self.conv1(feat, adj)
        if self.bias is not None:
            out += self.bias
        if self.activation is not None:
            out = self.activation(out)
        return out

class GCN_two(nn.Module):
    def __init__(self, input_dim, hid_dim1, hid_dim2, dropout=0., activation="relu"):
        super(GCN_two, self).__init__()
        self.conv1 = GCN_one(input_dim, hid_dim1)
        self.conv2 = GCN_one(hid_dim1, hid_dim2)

        self.dropout = dropout
        assert activation in ["relu", "leaky_relu", "elu"]
        self.activation = getattr(F, activation)

    def forward(self, feature, adj):
        x1 = self.activation(self.conv1(feature, adj))
        x1 = F.dropout(x1, p=self.dropout, training=self.training)
        x2 = self.conv2(x1, adj)
        return x2  # F.log_softmax(x2, dim=1)


class GCN_one(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True, activation=None):
        super(GCN_one, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.activation = activation
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, feat, adj):
        adj = adj.to_dense()
        feat = self.fc(feat)
        out = torch.spmm(adj, feat)
        if self.bias is not None:
            out += self.bias
        if self.activation is not None:
            out = self.activation(out)
        return out
