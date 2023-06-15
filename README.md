# Universal Graph Contrastive Learning with Novel Laplacian Perturbation [UAI2023] 
Model Overview
![ugcl_fig](https://github.com/twko05/UGCL/assets/80378163/c90b05b9-2780-467a-aae5-2a48f4305428)


# Code
An official code of [UGCL](https://github.com/twko05/UGCL/files/11755605/UGCL.pdf)


The codes are referenced from [MagNet](https://github.com/matthew-hirn/magnet) and [SDGNN](https://github.com/huangjunjie-cs/SiGAT/tree/master)
# Run
*--graph* gets structure perturbation ratio

*--laplacian* gets whether apply Laplacian perturbation

```Bash
python train.py --dataset BitCoinAlpha --graph 0.1 --laplacian 1
```
