# Lipschitz-HNN
## Introduction
Hyperbolic neural networks (HNNs) have shown promise in modeling data with non-Euclidean structures. However, their performance has often been compromised due to instability and a lack of robustness. In this work, we propose a novel approach to enhancing the stability of HNNs by conducting rigorous Lipschitz analysis. Our analysis covers both the Poincaré ball and the hyperboloid models, providing Lipschitz bounds for HNN layers. Importantly, this analysis yields valuable insights into features with unit norms and large norms within HNNs. Further, we introduce regularization methods based on the derived Lipschitz bounds and demonstrate consistent improvements in HNN stability against noisy perturbations. This part is the code where we perform Lipschitz regularized training for HNN using the Poincaré ball  ball model.
## Environment
The code is tested on Python 3.7.
* numpy==1.15.1
* scikit-learn==0.19.2
* torch==1.1.0
* torchvision==0.2.2
* networkx==2.1
## Training
Here we provide the scripts for different datasets.
* Texas (nodes represent web pages and edges represent hyperlinks)

```python train.py --task nc --dataset texas --model HNN --lr 0.01 --dim 16 --num-layers 2 --act None --bias 1 --dropout 0.5 --weight-decay 0 --manifold PoincareBall --log-freq 1 ```

* Wisconsin (nodes represent web pages and edges represent hyperlinks)
  
```python train.py --task nc --dataset wisconsin --model HNN --lr 0.01 --dim 16 --num-layers 2 --act None --bias 1 --dropout 0.5 --weight-decay 0 --manifold PoincareBall --log-freq 1```

* Chameleon (nodes represent Wikipedia pages and edges represent links between pairs of pages)
  
```python train.py --task nc --dataset chameleon --model HNN --lr 0.01 --dim 16 --num-layers 2 --act None --bias 1 --dropout 0.5 --weight-decay 0.001 --manifold PoincareBall --log-freq 1```

* Actor (nodes represent actors and two nodes are connected if they co-occur on the same Wikipedia page)
* 
```python train.py --task nc --dataset film --model HNN --dim 16 --lr 0.005 --num-layers 2 --act relu --bias 1 --dropout 0 --weight-decay 1e-4 --manifold PoincareBall --log-freq 1```

* Cora (nodes represent publications and edges represent citations)
```python train.py --task nc --dataset cora --model HNN --lr 0.01 --dim 16 --num-layers 2 --act None --bias 1 --dropout 0.2 --weight-decay 0.001 --manifold PoincareBall --log-freq 1```

* Pubmed (nodes represent publications and edges represent citations)
```python train.py --task nc --dataset pubmed --model HNN --lr 0.01 --dim 16 --num-layers 2 --act None --bias 1 --dropout 0.5 --weight-decay 0 --manifold PoincareBall --log-freq 1```
