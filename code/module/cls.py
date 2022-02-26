import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import GCN_two, GCN_two_pyg


class Classification(nn.Module):
    def __init__(self, num_feature, cls_hid_1, num_class, dropout, pyg):
        super(Classification, self).__init__()
        if pyg==False:
            self.encoder_v1 = GCN_two(num_feature, cls_hid_1, num_class, dropout)
            self.encoder_v2 = GCN_two(num_feature, cls_hid_1, num_class, dropout)
            self.encoder_v = GCN_two(num_feature, cls_hid_1, num_class, dropout)
        else:
            print("pyg")
            self.encoder_v1 = GCN_two_pyg(num_feature, cls_hid_1, num_class, dropout)
            self.encoder_v2 = GCN_two_pyg(num_feature, cls_hid_1, num_class, dropout)
            self.encoder_v = GCN_two_pyg(num_feature, cls_hid_1, num_class, dropout)

    def forward(self, feat, view, flag):
        if flag == "v1":
            prob = F.softmax(self.encoder_v1(feat, view), dim=1)
        elif flag == "v2":
            prob = F.softmax(self.encoder_v2(feat, view), dim=1)
        elif flag == "v":
            prob = F.softmax(self.encoder_v(feat, view), dim=1)
        return prob
