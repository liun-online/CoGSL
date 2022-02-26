import torch
import torch.nn as nn
import torch.nn.functional as F
from module.view_estimator import View_Estimator
from module.cls import Classification
from module.mi_nce import MI_NCE
from module.fusion import Fusion


class Cogsl(nn.Module):
    def __init__(self, num_feature, cls_hid_1, num_class, gen_hid, mi_hid_1,
                 com_lambda_v1, com_lambda_v2, lam, alpha, cls_dropout, ve_dropout, tau, pyg, big, batch, name):
        super(Cogsl, self).__init__()
        self.cls = Classification(num_feature, cls_hid_1, num_class, cls_dropout, pyg)
        self.ve = View_Estimator(num_feature, gen_hid, com_lambda_v1, com_lambda_v2, ve_dropout, pyg)
        self.mi = MI_NCE(num_feature, mi_hid_1, tau, pyg, big, batch)
        self.fusion = Fusion(lam, alpha, name)

    def get_view(self, data):
        new_v1, new_v2 = self.ve(data)
        return new_v1, new_v2

    def get_mi_loss(self, feat, views):
        mi_loss = self.mi(views, feat)
        return mi_loss

    def get_cls_loss(self, v1, v2, feat):
        prob_v1 = self.cls(feat, v1, "v1")
        prob_v2 = self.cls(feat, v2, "v2")
        logits_v1 = torch.log(prob_v1 + 1e-8)
        logits_v2 = torch.log(prob_v2 + 1e-8)
        return logits_v1, logits_v2, prob_v1, prob_v2

    def get_v_cls_loss(self, v, feat):
        logits = torch.log(self.cls(feat, v, "v") + 1e-8)
        return logits

    def get_fusion(self, v1, prob_v1, v2, prob_v2):
        v = self.fusion(v1, prob_v1, v2, prob_v2)
        return v
