import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos
import scipy.sparse as sp
from scipy.linalg import fractional_matrix_power, inv


def knn(feat, num_node, k, data_name, view_name):
    adj = np.zeros((num_node, num_node), dtype=np.int64)
    dist = cos(feat)
    col = np.argpartition(dist, -(k + 1), axis=1)[:, -(k + 1):].flatten()
    adj[np.arange(dataset.num_node).repeat(k + 1), col] = 1  
    adj = sp.coo_matrix(adj)
    sp.save_npz("./"+data_name+"/"+view_name+"_knn.npz", adj)


def adj(adj, data_name, view_name):
    adj = sp.coo_matrix(adj)
    sp.save_npz("./"+data_name+"/"+view_name+"_adj.npz", adj)


def diff(adj, alpha, data_name, view_name):   
    d = np.diag(np.sum(adj, 1))                                    
    dinv = fractional_matrix_power(d, -0.5)                       
    at = np.matmul(np.matmul(dinv, adj), dinv)                      
    adj = alpha * inv((np.eye(adj.shape[0]) - (1 - alpha) * at))   
    adj = sp.coo_matrix(adj)
    sp.save_npz("./"+data_name+"/"+view_name+"_diff.npz", adj)

data_name = "wine"
view_name = "v1"  # v1 or v2
view_type = "knn"  # knn adj diff

adj = sp.load("./"+data_name+"/ori_adj.npz")
num_node = adj.shape[0]
feat = sp.load("./"+data_name+"/feat.npz")
a = adj.A
if a[0, 0] == 0:
    a += np.eye(dataset.num_node)
    print("self-loop!")
adj = a
if view_type == "knn":  # set k
    knn(feat, num_node, k, data_name, view_name)
elif view_type == "adj":
    adj(adj, data_name, view_name)
elif view_type == "diff":  # set alpha: 0~1
    diff(adj, alpha, data_name, view_name)
