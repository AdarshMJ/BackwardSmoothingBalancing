# BackwardSmoothingBalancing



![balancedbackward](https://github.com/AdarshMJ/BackwardSmoothingBalancing/blob/main/balancedbackward.jpg)





## TL;DR of the paper

1. New paper [Backward Oversmoothing: why is it hard to train deep Graph Neural Networks?](https://arxiv.org/abs/2505.16736) explores why it is hard to train deeper GNNs from an optimization perspective.
2. Over-smoothing is a well-known phenomenon in GNNs, where node features become indistinguishable as the number of layers/rounds of aggregation increases if the weights of the model are bounded. If the weights are sufficiently large, then the GNN should not over-smooth. But this doesnt happen in practice.
3. This is because of "backward oversmoothing", that is, the errors that are propagated during gradient descent might also be subject to over-smoothing.

## My hypothesis 
1. Find better way to initialize the GNN model weights. This work for instance - [Are GATs Out of Balance?](https://arxiv.org/pdf/2310.07235) proposes a way to initialize the model weights (balanced+orthogonal initialization).
2. By initializing the weights in balance, we can mitigate backward smoothing! Since balancing the weights lead to effective gradient flow and thus curbs grandient vanishing consequently tackling backward oversmoothing.

## How to use the code
### Pre-requisities
```Python
1. Pytorch
2. Pytorch-Geometric
```
### Dataset
We use the synthetic-cora dataset proposed in [Beyond Homophily in Graph Neural Networks:
Current Limitations and Effective Designs](https://arxiv.org/pdf/2006.11468), we provide 5 datasets with varying levels of homophily level.

### Run Code
```python
python backwardsmoothing.py
```




   
