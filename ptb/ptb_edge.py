import numpy as np
import scipy.sparse as sp

data_name = "breast_cancer"
view = "../dataset/"+data_name+"/ori_adj.npz"

adj = sp.load_npz(view).A
adj_fan = 1 - adj
adj_ = sp.triu(sp.coo_matrix(adj), 1)
adj_fan_ = sp.triu(sp.coo_matrix(adj_fan), 1)

adj_cand = np.array(adj_.nonzero())
adj_fan_cand = np.array(adj_fan_.nonzero())

dele = [0.05,0.1,0.15]
add = [0.25,0.5,0.75]
for ra in range(3):
    dele_num = int((1-dele[ra]) * adj_cand.shape[1])
    print(dele_num)    
    add_num = int(add[ra] * adj_cand.shape[1])
    print(add_num)
    
    adj_sele = np.random.choice(np.arange(adj_cand.shape[1]), dele_num, replace = False)
    adj_sele = adj_cand[:, adj_sele]
    adj_new = sp.coo_matrix((np.ones(adj_sele.shape[1]),(adj_sele[0, :], adj_sele[1, :])), shape = adj_.shape)
    adj_new = adj_new + adj_new.T + sp.eye(adj_new.shape[0])   
    sp.save_npz(data_name+"_dele_"+str(dele[ra])+".npz", adj_new)
    
    adj_sele = np.random.choice(np.arange(adj_fan_cand.shape[1]), add_num, replace = False)
    adj_sele = adj_fan_cand[:, adj_sele]
    adj_sele = np.hstack([adj_sele, adj_cand])    
    adj_new = sp.coo_matrix((np.ones(adj_sele.shape[1]),(adj_sele[0, :], adj_sele[1, :])), shape = adj_.shape)
    adj_new = adj_new + adj_new.T + sp.eye(adj_new.shape[0])
    sp.save_npz(data_name+"_add_"+str(add[ra])+".npz", adj_new)
    