import numpy as np
import scipy.sparse as sp
import torch


def get_khop_indices(k, view):
    view = (view.A > 0).astype("int32")
    view_ = view
    for i in range(1, k):
        view_ = (np.matmul(view_, view.T)>0).astype("int32")
    view_ = torch.tensor(view_).to_sparse()
    return view_.indices()
    
def topk(k, adj):
    pos = np.zeros(adj.shape)
    for i in range(len(adj)):
      one = adj[i].nonzero()[0]
      if len(one)>k:
        oo = np.argsort(-adj[i, one])
        sele = one[oo[:k]]
        pos[i, sele] = adj[i, sele]
      else:
        pos[i, one] = adj[i, one]
    return pos

#####################
## get k-hop scope ##
## take citeseer   ##
#####################
adj = sp.load_npz("./citeseer/v1_adj.npz")
indice = get_khop_indices(2, adj)
torch.save(indice, "./citeseer/v1_2.pt")

#####################
## get top-k scope ##
## take citeseer   ##
#####################
adj = sp.load_npz("./citeseer/v2_diff.npz")
kn = topk(40, adj)
kn = sp.coo_matrix(kn)
indice = get_khop_indices(1, kn)
torch.save(indice, "./citeseer/v2_40.pt")
