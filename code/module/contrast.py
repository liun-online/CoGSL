import torch
import torch.nn as nn


class Contrast:
    def __init__(self, tau):
        self.tau = tau

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def cal(self, z1_proj, z2_proj):
        matrix_z1z2 = self.sim(z1_proj, z2_proj)
        matrix_z2z1 = matrix_z1z2.t()

        matrix_z1z2 = matrix_z1z2 / (torch.sum(matrix_z1z2, dim=1).view(-1, 1) + 1e-8)
        lori_v1v2 = -torch.log(matrix_z1z2.diag()+1e-8).mean()

        matrix_z2z1 = matrix_z2z1 / (torch.sum(matrix_z2z1, dim=1).view(-1, 1) + 1e-8)
        lori_v2v1 = -torch.log(matrix_z2z1.diag()+1e-8).mean()
        return (lori_v1v2 + lori_v2v1) / 2
