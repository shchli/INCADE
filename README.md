# Navigating the Implicit Map: Community-Aware Disentangled Experts for Multi-Modal Knowledge Graph Completion


##  Dependencies
- Python==3.9
- numpy==1.24.2
- scikit_learn==1.2.2
- torch==2.4.0
- dgl = 2.4.0
- tqdm==4.64.1


## Train and Evaluation


```bash
nohup python train.py --cuda 0 --lr 0.001 --lambda1 0.02 --lambda2 0.05 --dim 200 --dataset MKG-W --epochs 2000 --group_num 4 --gtype gmm --rgcngraph True > log.txt &

nohup python train.py --cuda 0 --lr 0.001 --lambda1 0.02 --lambda2 0.05 --dim 200 --dataset DB15K --epochs 2000 --group_num 4 --gtype gmm --rgcngraph True > log.txt &
```




