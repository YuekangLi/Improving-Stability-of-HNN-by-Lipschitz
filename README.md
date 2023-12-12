# Lipschitz-HNN
## Introduction
Hyperbolic neural networks (HNNs) have shown promise in modeling data with non-Euclidean structures. However, their performance has often been compromised due to instability and a lack of robustness. In this work, we propose a novel approach to enhancing the stability of HNNs by conducting rigorous Lipschitz analysis. Our analysis covers both the Poincaré ball and the hyperboloid models, providing Lipschitz bounds for HNN layers. Importantly, this analysis yields valuable insights into features with unit norms and large norms within HNNs. Further, we introduce regularization methods based on the derived Lipschitz bounds and demonstrate consistent improvements in HNN stability against noisy perturbations. This part is the code where we perform Lipschitz regularized training for HNN using the Poincaré ball  ball model.
## Environment
numpy==1.16.2
numpy=1.16.2
scikit-learn==0.20.3
torch==1.1.0
torchvision==0.2.2
networkx==2.2
## Training
**hnn_nc_cornell_poincare** 

python [train.py](http://train.py/) --task nc --dataset cornell --model HNN --lr 0.01 --dim 16 --num-layers 2 --act None --bias 1 --dropout 0.5 --weight-decay 0 --manifold PoincareBall --log-freq 5 --cuda 0 >log_result.file 2>&1

**hnn_nc_cornell_hyperboloid**

python [train.py](http://train.py/) --task nc --dataset cornell --model HNN --lr 0.01 --dim 16 --num-layers 2 --act None --bias 1 --dropout 0.5 --weight-decay 0 --manifold Hyperboloid --log-freq 5 --cuda 2 >log_result.file 2>&1

**hnn_nc_texas_poincare** 

python [train.py](http://train.py/) --task nc --dataset texas --model HNN --lr 0.01 --dim 16 --num-layers 2 --act None --bias 1 --dropout 0.5 --weight-decay 0 --manifold PoincareBall --log-freq 5 --cuda 0 >log_result.file 2>&1

**hnn_nc_texas_hyperboloid**

python [train.py](http://train.py/) --task nc --dataset texas --model HNN --lr 0.01 --dim 16 --num-layers 2 --act None --bias 1 --dropout 0.5 --weight-decay 0 --manifold Hyperboloid --log-freq 5 --cuda 2 >log_result.file 2>&1

**hnn_nc_wisconsin_poincare** 

python [train.py](http://train.py/) --task nc --dataset wisconsin --model HNN --lr 0.01 --dim 16 --num-layers 2 --act None --bias 1 --dropout 0.5 --weight-decay 0 --manifold PoincareBall --log-freq 5 --cuda 0 >log_result.file 2>&1

**hnn_nc_wisconsin_hyperboloid**

python [train.py](http://train.py/) --task nc --dataset wisconsin --model HNN --lr 0.01 --dim 16 --num-layers 2 --act None --bias 1 --dropout 0.5 --weight-decay 0 --manifold Hyperboloid --log-freq 5 --cuda 2 >log_result.file 2>&1

**hnn_nc_chameleon_poincare** 

python [train.py](http://train.py/) --task nc --dataset chameleon --model HNN --lr 0.01 --dim 16 --num-layers 2 --act None --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall --log-freq 5 --cuda 0 --c 1 >log_result.file 2>&1

python [train.py](http://train.py/) --task nc --dataset chameleon --model HNN --lr 0.01 --dim 16 --num-layers 2 --act None --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold Hyperboloid --log-freq 5 --cuda 2 --c 1 >log_result.file 2>&1

python [train.py](http://train.py/) --task nc --dataset chameleon --model HNN --lr 0.01 --dim 16 --num-layers 2 --act None --bias 1 --dropout 0.5 --weight-decay 0 --manifold Hyperboloid --log-freq 5 --cuda 0 >log_result.file 2>&1

**hnn_nc_squirrel_poincare** 

python [train.py](http://train.py/) --task nc --dataset squirrel --model HNN --dim 16 --lr 0.005 --num-layers 3 --act relu --bias 1 --dropout 0.2 --weight-decay 1e-4 --manifold PoincareBall --log-freq 5 --patience 1500 --cuda 0 >log_result.file 2>&1

**hnn_nc_squirrel_hyperbploid** 

python [train.py](http://train.py/) --task nc --dataset squirrel --model HNN --dim 16 --lr 0.005 --num-layers 3 --act relu --bias 1 --dropout 0.2 --weight-decay 1e-4 --manifold Hyperboloid --log-freq 5 --patience 1500 --cuda 0 >log_result.file 2>&1

**hnn_nc_actor_poincare** 

python [train.py](http://train.py/) --task nc --dataset film --model HNN --dim 16 --lr 0.005 --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 1e-4 --manifold PoincareBall --log-freq 5 --patience 1500 --cuda 0 >log_result.file 2>&1

**hnn_nc_actor_hyperboloid** 

python [train.py](http://train.py/) --task nc --dataset film --model HNN --dim 16 --lr 0.005 --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 1e-4 --manifold Hyperboloid --log-freq 5 --patience 1500 --cuda 0 >log_result.file 2>&1

**hnn_nc_cora_poincare**

python [train.py](http://train.py/) --task nc --dataset cora --model HNN --lr 0.01 --dim 16 --num-layers 2 --act None --bias 1 --dropout 0.2 --weight-decay 0.001 --manifold PoincareBall --log-freq 5 --cuda 0 --c 1 >log_result.file 2>&1

**hnn_nc_cora_hyperboloid**

python [train.py](http://train.py/) --task nc --dataset cora --model HNN --lr 0.01 --dim 16 --num-layers 2 --act None --bias 1 --dropout 0.5 --weight-decay 0 --manifold Hyperboloid --log-freq 5 --cuda 0 >log_result.file 2>&1

**hnn_nc_pubmed_poincare**

python [train.py](http://train.py/) --task nc --dataset pubmed --model HNN --lr 0.01 --dim 16 --num-layers 2 --act None --bias 1 --dropout 0.5 --weight-decay 0 --manifold PoincareBall --log-freq 5 --cuda 0 >log_result.file 2>&1

**hnn_nc_pubmed_hyperbolic**

python [train.py](http://train.py/) --task nc --dataset pubmed --model HNN --lr 0.01 --dim 16 --num-layers 2 --act None --bias 1 --dropout 0.5 --weight-decay 0 --manifold Hyperboloid --log-freq 5 --cuda 0 >log_result.file 2>&1
