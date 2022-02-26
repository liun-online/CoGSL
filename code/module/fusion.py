import torch
import torch.nn as nn
import torch.nn.functional as F


class Fusion(nn.Module):
    def __init__(self, lam, alpha, name):
        super(Fusion, self).__init__()
        self.lam = lam
        self.alpha = alpha
        self.name = name

    def get_weight(self, prob):
        out, _ = prob.topk(2, dim=1, largest=True, sorted=True)
        fir = out[:, 0]
        sec = out[:, 1]
        w = torch.exp(self.alpha*(self.lam*torch.log(fir+1e-8) + (1-self.lam)*torch.log(fir-sec+1e-8)))
        return w

    def forward(self, v1, prob_v1, v2, prob_v2):
        w_v1 = self.get_weight(prob_v1)
        w_v2 = self.get_weight(prob_v2)
        beta_v1 = w_v1 / (w_v1 + w_v2)
        beta_v2 = w_v2 / (w_v1 + w_v2)
        if self.name not in ["citeseer", "digits", "polblogs"]:
            beta_v1 = beta_v1.diag().to_sparse()
            beta_v2 = beta_v2.diag().to_sparse()
            v = torch.sparse.mm(beta_v1, v1) + torch.sparse.mm(beta_v2, v2)
            return v
        else :
            beta_v1 = beta_v1.unsqueeze(1)
            beta_v2 = beta_v2.unsqueeze(1)
            v = beta_v1 * v1.to_dense() + beta_v2 * v2.to_dense()
            return v.to_sparse()
