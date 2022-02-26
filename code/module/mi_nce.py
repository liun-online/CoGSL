import torch
import torch.nn as nn
from .base import GCN_one, GCN_one_pyg
from .contrast import Contrast
import numpy as np


class MI_NCE(nn.Module):
    def __init__(self, num_feature, mi_hid_1, tau, pyg, big, batch):
        super(MI_NCE, self).__init__()
        if pyg == False:
            self.gcn = GCN_one(num_feature, mi_hid_1, activation=nn.PReLU())
            self.gcn1 = GCN_one(num_feature, mi_hid_1, activation=nn.PReLU())
            self.gcn2 = GCN_one(num_feature, mi_hid_1, activation=nn.PReLU())
        else:
            print("pyg")
            self.gcn = GCN_one_pyg(num_feature, mi_hid_1, activation=nn.PReLU())
            self.gcn1 = GCN_one_pyg(num_feature, mi_hid_1, activation=nn.PReLU())
            self.gcn2 = GCN_one_pyg(num_feature, mi_hid_1, activation=nn.PReLU())
        self.proj = nn.Sequential(
            nn.Linear(mi_hid_1, mi_hid_1),
            nn.ELU(),
            nn.Linear(mi_hid_1, mi_hid_1)
        )
        self.con = Contrast(tau)
        self.big = big
        self.batch = batch

    def forward(self, views, feat):
        v_emb = self.proj(self.gcn(feat, views[0]))
        v1_emb = self.proj(self.gcn1(feat, views[1]))
        v2_emb = self.proj(self.gcn2(feat, views[2]))
        # if dataset is so big, we will randomly sample part of nodes to perform MI estimation
        if self.big == True:
            idx = np.random.choice(feat.shape[0], self.batch, replace=False)
            idx.sort()
            v_emb = v_emb[idx]
            v1_emb = v1_emb[idx]
            v2_emb = v2_emb[idx]
            
        vv1 = self.con.cal(v_emb, v1_emb)
        vv2 = self.con.cal(v_emb, v2_emb)
        v1v2 = self.con.cal(v1_emb, v2_emb)

        return vv1, vv2, v1v2
