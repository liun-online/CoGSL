# CoGSL
Compact Graph Structure Learning via Mutual Information Compression (TheWebConf 2022)\
Paper Link: https://arxiv.org/abs/2201.05540

# Environment Settings
```
python==3.8.5
numpy==1.19.2
scikit_learn==1.0.2
scipy==1.6.2
torch==1.9.0
torch_geometric==1.7.2
torch_sparse==0.6.11
```
GPU: GeForce RTX 3090 \
CPU: Intel(R) Xeon(R) Silver 4210 CPU @ 2.20GHz

# Usage
## Datasets
1. Get the datasets from https://pan.baidu.com/s/1_tT6CSiqCkpdoFonJrpU0g, with password: 21pa
2. Unzip them into ./dataset/
## Attacked Datasets
1. Get the datasets from https://pan.baidu.com/s/1u7ep2gPAD_iocuDJF4Fsdg, with password: gm7w
2. Unzip them into ./ptb/

After dealing with datasets, you can go into ./code/, and then run the following command:
```
python main.py wine --gpu=0
```
where "wine" can be replaced by "breast_cancer", "digits", "polblogs", "citeseer", "wikics" or "ms". \
If you futher want to reproduce the results under attacks, please use the following command:
```
python main.py breast_cancer --gpu=0 --dele --flag=1 --ratio=0.05
```
where datasets∈["breast_cancer", "citeseer", "wikics"], --dele can be replaced by --add, --flag∈[1, 2, 3], 1 or 2 means the 1st or 2nd view is attacked, and 3 means both of them are attacked. --ratio is in [0.05, 0.1, 0.15] under --dele, while in [0.25, 0.5, 0.75] under --add

# Cite

# Contact
If you have any questions, please feel free to contact me with {nianliu@bupt.edu.cn}
