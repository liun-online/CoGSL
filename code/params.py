import argparse
import sys

argv = sys.argv
dataset = argv[1]


def wine_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset', type=str, default="wine")
    parser.add_argument('--batch', type=int, default=0)
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=4)    

    parser.add_argument('--name_view1', type=str, default="v1_knn")
    parser.add_argument('--name_view2', type=str, default="v2_diff")
    parser.add_argument('--indice_view1', type=str, default="v1_2")
    parser.add_argument('--indice_view2', type=str, default="v2_40")
    
    parser.add_argument('--cls_hid_1', type=int, default=16)
    
    ## gen 
    parser.add_argument('--com_lambda_v1', type=float, default=0.5)  
    parser.add_argument('--com_lambda_v2', type=float, default=0.5)  
    parser.add_argument('--gen_hid', type=int, default=64)
    
    ## fusion
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.1)
    
    ## mi
    parser.add_argument('--mi_hid_1', type=int, default=128)
    
    ## optimizer
    parser.add_argument('--cls_lr', type=float, default=0.01)
    parser.add_argument('--cls_weight_decay', type=float, default=5e-4)
    parser.add_argument('--cls_dropout', type=float, default=0.5)

    parser.add_argument('--ve_lr', type=float, default=0.001)
    parser.add_argument('--ve_weight_decay', type=float, default=0.)
    parser.add_argument('--ve_dropout', type=float, default=0.8)

    parser.add_argument('--mi_lr', type=float, default=0.01)
    parser.add_argument('--mi_weight_decay', type=float, default=0)
    
    ## iter
    parser.add_argument('--main_epoch', type=int, default=100)
    parser.add_argument('--inner_ne_epoch', type=int, default=1)  
    parser.add_argument('--inner_cls_epoch', type=int, default=1) 
    parser.add_argument('--inner_mi_epoch', type=int, default=5) 
    parser.add_argument('--temp_r', type=float, default=1e-3)
    
    ## coe
    parser.add_argument('--cls_coe', type=float, default=0.3)
    parser.add_argument('--mi_coe', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=0.5)
    
    ## ptb
    parser.add_argument('--add', action="store_true")
    parser.add_argument('--dele', action="store_true")
    parser.add_argument('--ptb_feat', action='store_true')
    
    parser.add_argument('--ratio', type=float,default=0.)
    parser.add_argument('--flag', type=int, default=1) 
    #####################################
    
    args, _ = parser.parse_known_args()
    return args

def breast_cancer_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset', type=str, default="breast_cancer")
    parser.add_argument('--batch', type=int, default=0)
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=6)    

    parser.add_argument('--name_view1', type=str, default="v1_knn")
    parser.add_argument('--name_view2', type=str, default="v2_diff")
    parser.add_argument('--indice_view1', type=str, default="v1_1")
    parser.add_argument('--indice_view2', type=str, default="v2_300")
    
    parser.add_argument('--cls_hid_1', type=int, default=16)
    
    ## gen 
    parser.add_argument('--com_lambda_v1', type=float, default=0.1)  
    parser.add_argument('--com_lambda_v2', type=float, default=0.1)  
    parser.add_argument('--gen_hid', type=int, default=64)
    
    ## fusion
    parser.add_argument('--lam', type=float, default=0.9)
    parser.add_argument('--alpha', type=float, default=0.1)
    
    ## mi
    parser.add_argument('--mi_hid_1', type=int, default=128)
    
    ## optimizer
    parser.add_argument('--cls_lr', type=float, default=0.01)
    parser.add_argument('--cls_weight_decay', type=float, default=5e-4)
    parser.add_argument('--cls_dropout', type=float, default=0.5)

    parser.add_argument('--ve_lr', type=float, default=0.1)
    parser.add_argument('--ve_weight_decay', type=float, default=0.)
    parser.add_argument('--ve_dropout', type=float, default=0.5)

    parser.add_argument('--mi_lr', type=float, default=0.01)
    parser.add_argument('--mi_weight_decay', type=float, default=0)
    
    ## iter
    parser.add_argument('--main_epoch', type=int, default=150)
    parser.add_argument('--inner_ne_epoch', type=int, default=1)  
    parser.add_argument('--inner_cls_epoch', type=int, default=1) 
    parser.add_argument('--inner_mi_epoch', type=int, default=5) 
    parser.add_argument('--temp_r', type=float, default=1e-3)
    
    ## coe
    parser.add_argument('--cls_coe', type=float, default=0.3)
    parser.add_argument('--mi_coe', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=0.5)
    
    ## ptb
    parser.add_argument('--add', action="store_true")
    parser.add_argument('--dele', action="store_true")
    parser.add_argument('--ptb_feat', action='store_true')
    
    parser.add_argument('--ratio', type=float,default=0.)
    parser.add_argument('--flag', type=int, default=1) 
    #####################################
    
    args, _ = parser.parse_known_args()
    return args

def digits_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset', type=str, default="digits")
    parser.add_argument('--batch', type=int, default=0)
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2)    

    parser.add_argument('--name_view1', type=str, default="v1_adj")
    parser.add_argument('--name_view2', type=str, default="v2_diff")
    parser.add_argument('--indice_view1', type=str, default="v1_1")
    parser.add_argument('--indice_view2', type=str, default="v2_100")
    
    parser.add_argument('--cls_hid_1', type=int, default=16)
    
    ## gen 
    parser.add_argument('--com_lambda_v1', type=float, default=0.5)  
    parser.add_argument('--com_lambda_v2', type=float, default=0.5)  
    parser.add_argument('--gen_hid', type=int, default=32)
    
    ## fusion
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.1)
    
    ## mi
    parser.add_argument('--mi_hid_1', type=int, default=128)
    
    ## optimizer
    parser.add_argument('--cls_lr', type=float, default=0.01)
    parser.add_argument('--cls_weight_decay', type=float, default=5e-4)
    parser.add_argument('--cls_dropout', type=float, default=0.5)

    parser.add_argument('--ve_lr', type=float, default=0.01)
    parser.add_argument('--ve_weight_decay', type=float, default=0.)
    parser.add_argument('--ve_dropout', type=float, default=0.5)

    parser.add_argument('--mi_lr', type=float, default=0.01)
    parser.add_argument('--mi_weight_decay', type=float, default=0)
    
    ## iter
    parser.add_argument('--main_epoch', type=int, default=200)
    parser.add_argument('--inner_ne_epoch', type=int, default=1)  
    parser.add_argument('--inner_cls_epoch', type=int, default=10) 
    parser.add_argument('--inner_mi_epoch', type=int, default=10) 
    parser.add_argument('--temp_r', type=float, default=1e-4)
    
    ## coe
    parser.add_argument('--cls_coe', type=float, default=0.3)
    parser.add_argument('--mi_coe', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=0.2)
    
    ## ptb
    parser.add_argument('--add', action="store_true")
    parser.add_argument('--dele', action="store_true")
    parser.add_argument('--ptb_feat', action='store_true')
    
    parser.add_argument('--ratio', type=float,default=0.)
    parser.add_argument('--flag', type=int, default=1) 
    #####################################
    
    args, _ = parser.parse_known_args()
    return args
    
def polblogs_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset', type=str, default="polblogs")
    parser.add_argument('--batch', type=int, default=0)
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)    

    parser.add_argument('--name_view1', type=str, default="v1_adj")
    parser.add_argument('--name_view2', type=str, default="v2_diff")
    parser.add_argument('--indice_view1', type=str, default="v1_1")
    parser.add_argument('--indice_view2', type=str, default="v2_500")
    
    parser.add_argument('--cls_hid_1', type=int, default=16)
    
    ## gen 
    parser.add_argument('--com_lambda_v1', type=float, default=0.1)  
    parser.add_argument('--com_lambda_v2', type=float, default=1.0)  
    parser.add_argument('--gen_hid', type=int, default=64)
    
    ## fusion
    parser.add_argument('--lam', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.1)
    
    ## mi
    parser.add_argument('--mi_hid_1', type=int, default=128)
    
    ## optimizer
    parser.add_argument('--cls_lr', type=float, default=0.01)
    parser.add_argument('--cls_weight_decay', type=float, default=5e-4)
    parser.add_argument('--cls_dropout', type=float, default=0.5)

    parser.add_argument('--ve_lr', type=float, default=0.1)
    parser.add_argument('--ve_weight_decay', type=float, default=0.)
    parser.add_argument('--ve_dropout', type=float, default=0.8)

    parser.add_argument('--mi_lr', type=float, default=0.01)
    parser.add_argument('--mi_weight_decay', type=float, default=0)
    
    ## iter
    parser.add_argument('--main_epoch', type=int, default=150)
    parser.add_argument('--inner_ne_epoch', type=int, default=1)  
    parser.add_argument('--inner_cls_epoch', type=int, default=5) 
    parser.add_argument('--inner_mi_epoch', type=int, default=5) 
    parser.add_argument('--temp_r', type=float, default=1e-4)
    
    ## coe
    parser.add_argument('--cls_coe', type=float, default=0.3)
    parser.add_argument('--mi_coe', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=0.8)
    
    ## ptb
    parser.add_argument('--add', action="store_true")
    parser.add_argument('--dele', action="store_true")
    parser.add_argument('--ptb_feat', action='store_true')
    
    parser.add_argument('--ratio', type=float,default=0.)
    parser.add_argument('--flag', type=int, default=1) 
    #####################################
    
    args, _ = parser.parse_known_args()
    return args
    
def citeseer_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset', type=str, default="citeseer")
    parser.add_argument('--batch', type=int, default=0)
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=3)    

    parser.add_argument('--name_view1', type=str, default="v1_adj")
    parser.add_argument('--name_view2', type=str, default="v2_diff")
    parser.add_argument('--indice_view1', type=str, default="v1_2")
    parser.add_argument('--indice_view2', type=str, default="v2_40")
    
    parser.add_argument('--cls_hid_1', type=int, default=16)
    
    ## gen 
    parser.add_argument('--com_lambda_v1', type=float, default=0.1)  
    parser.add_argument('--com_lambda_v2', type=float, default=0.1)  
    parser.add_argument('--gen_hid', type=int, default=32)
    
    ## fusion
    parser.add_argument('--lam', type=float, default=0.5)
    parser.add_argument('--alpha', type=float, default=0.1)
    
    ## mi
    parser.add_argument('--mi_hid_1', type=int, default=128)
    
    ## optimizer
    parser.add_argument('--cls_lr', type=float, default=0.01)
    parser.add_argument('--cls_weight_decay', type=float, default=5e-4)
    parser.add_argument('--cls_dropout', type=float, default=0.5)

    parser.add_argument('--ve_lr', type=float, default=0.001)
    parser.add_argument('--ve_weight_decay', type=float, default=0.)
    parser.add_argument('--ve_dropout', type=float, default=0.2)

    parser.add_argument('--mi_lr', type=float, default=0.01)
    parser.add_argument('--mi_weight_decay', type=float, default=0)
    
    ## iter
    parser.add_argument('--main_epoch', type=int, default=200)
    parser.add_argument('--inner_ne_epoch', type=int, default=5)  
    parser.add_argument('--inner_cls_epoch', type=int, default=5) 
    parser.add_argument('--inner_mi_epoch', type=int, default=10) 
    parser.add_argument('--temp_r', type=float, default=1e-4)
    
    ## coe
    parser.add_argument('--cls_coe', type=float, default=0.3)
    parser.add_argument('--mi_coe', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=0.8)
    
    ## ptb
    parser.add_argument('--add', action="store_true")
    parser.add_argument('--dele', action="store_true")
    parser.add_argument('--ptb_feat', action='store_true')
    
    parser.add_argument('--ratio', type=float,default=0.)
    parser.add_argument('--flag', type=int, default=1) 
    #####################################
    
    args, _ = parser.parse_known_args()
    return args
   
def wikics_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset', type=str, default="wikics")
    parser.add_argument('--batch', type=int, default=1000)
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=6)    

    parser.add_argument('--name_view1', type=str, default="v1_adj")
    parser.add_argument('--name_view2', type=str, default="v2_sub")
    parser.add_argument('--indice_view1', type=str, default="v1_1")
    parser.add_argument('--indice_view2', type=str, default="v2_1")
    
    parser.add_argument('--cls_hid_1', type=int, default=16)
    
    ## gen 
    parser.add_argument('--com_lambda_v1', type=float, default=0.5)  
    parser.add_argument('--com_lambda_v2', type=float, default=0.5)  
    parser.add_argument('--gen_hid', type=int, default=16)
    
    ## fusion
    parser.add_argument('--lam', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.1)
    
    ## mi
    parser.add_argument('--mi_hid_1', type=int, default=32)
    
    ## optimizer
    parser.add_argument('--cls_lr', type=float, default=0.01)
    parser.add_argument('--cls_weight_decay', type=float, default=5e-4)
    parser.add_argument('--cls_dropout', type=float, default=0.5)

    parser.add_argument('--ve_lr', type=float, default=0.01)
    parser.add_argument('--ve_weight_decay', type=float, default=0)
    parser.add_argument('--ve_dropout', type=float, default=0.2)

    parser.add_argument('--mi_lr', type=float, default=0.01)
    parser.add_argument('--mi_weight_decay', type=float, default=0)
    
    ## iter
    parser.add_argument('--main_epoch', type=int, default=200)
    parser.add_argument('--inner_ne_epoch', type=int, default=1)  
    parser.add_argument('--inner_cls_epoch', type=int, default=1) 
    parser.add_argument('--inner_mi_epoch', type=int, default=5) 
    parser.add_argument('--temp_r', type=float, default=1e-3)
    
    ## coe
    parser.add_argument('--cls_coe', type=float, default=0.3)
    parser.add_argument('--mi_coe', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=0.5)
    
    ## ptb
    parser.add_argument('--add', action="store_true")
    parser.add_argument('--dele', action="store_true")
    parser.add_argument('--ptb_feat', action='store_true')
    
    parser.add_argument('--ratio', type=float,default=0.)
    parser.add_argument('--flag', type=int, default=1) 
    #####################################
    
    args, _ = parser.parse_known_args()
    return args

def ms_params():
    parser = argparse.ArgumentParser()
    #####################################
    ## basic info
    parser.add_argument('--dataset', type=str, default="ms")
    parser.add_argument('--batch', type=int, default=1000)
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--seed', type=int, default=4)    

    parser.add_argument('--name_view1', type=str, default="v1_adj")
    parser.add_argument('--name_view2', type=str, default="v2_sub")
    parser.add_argument('--indice_view1', type=str, default="v1_1")
    parser.add_argument('--indice_view2', type=str, default="v2_1")
    
    parser.add_argument('--cls_hid_1', type=int, default=16)
    
    ## gen 
    parser.add_argument('--com_lambda_v1', type=float, default=0.5)  
    parser.add_argument('--com_lambda_v2', type=float, default=0.5)  
    parser.add_argument('--gen_hid', type=int, default=32)
    
    ## fusion
    parser.add_argument('--lam', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=1.0)
    
    ## mi
    parser.add_argument('--mi_hid_1', type=int, default=256)
    
    ## optimizer
    parser.add_argument('--cls_lr', type=float, default=0.01)
    parser.add_argument('--cls_weight_decay', type=float, default=5e-4)
    parser.add_argument('--cls_dropout', type=float, default=0.5)

    parser.add_argument('--ve_lr', type=float, default=0.0001)
    parser.add_argument('--ve_weight_decay', type=float, default=1e-10)
    parser.add_argument('--ve_dropout', type=float, default=0.8)

    parser.add_argument('--mi_lr', type=float, default=0.01)
    parser.add_argument('--mi_weight_decay', type=float, default=0)
    
    ## iter
    parser.add_argument('--main_epoch', type=int, default=200)
    parser.add_argument('--inner_ne_epoch', type=int, default=1)  
    parser.add_argument('--inner_cls_epoch', type=int, default=15) 
    parser.add_argument('--inner_mi_epoch', type=int, default=10) 
    parser.add_argument('--temp_r', type=float, default=1e-4)
    
    ## coe
    parser.add_argument('--cls_coe', type=float, default=0.3)
    parser.add_argument('--mi_coe', type=float, default=0.3)
    parser.add_argument('--tau', type=float, default=0.5)
    
    ## ptb
    parser.add_argument('--add', action="store_true")
    parser.add_argument('--dele', action="store_true")
    parser.add_argument('--ptb_feat', action='store_true')
    
    parser.add_argument('--ratio', type=float,default=0.)
    parser.add_argument('--flag', type=int, default=1) 
    #####################################
    
    args, _ = parser.parse_known_args()
    return args

def set_params():
    if dataset == "wine":
        args = wine_params()
        args.pyg = False
        args.big = False
    elif dataset == "breast_cancer":
        args = breast_cancer_params()
        args.pyg = False
        args.big = False
    elif dataset == "digits":
        args = digits_params()
        args.pyg = False
        args.big = False
    elif dataset == "polblogs":
        args = polblogs_params()
        args.pyg = False
        args.big = False
    elif dataset == "citeseer":
        args = citeseer_params()
        args.pyg = False
        args.big = False
    elif dataset == "wikics":
        args = wikics_params()
        args.pyg = True
        args.big = True
    elif dataset == "ms":
        args = ms_params()
        args.pyg = True
        args.big = True
    return args
