# Deep-generative-model-project
Final project of deep-generative-model in Spring 2020

This is the code for reproducing the experimental results in paper [A Closer Look at the Optimization Landscapes of Generative Adversarial Networks](https://arxiv.org/abs/1906.04848), Hugo Berard, Gauthier Gidel,  Amjad Almahairi, Pascal Vincent, Simon Lacoste-Julien, 2019.

For any questions regarding the code please contact Hugo Berard (berard.hugo@gmail.com).

## Running the Code
Two datasets: Mixture of Gaussian and MNIST are available.
Four objective loss: NSGAN, LSGAN, WGAN, WGAN-GP are available.

To run the code for the Mixture of Gaussian experiment:
First, get Mixture of Gaussian dataset:
`python data_gen.py`
  
Second, train for the Mixture of Gaussian experiment:
`python gaussian/train.py --epochs 100000 --en_dim 1 --save_dir results`

To run the code for the MNIST experiment (NSGAN, LSGAN, WGAN, WGAN-GP are available):
`python mnist/train.py --epochs 50 --en_dim 5 --save_dir results --loss nsgan`

The visualization of the results can be done with `plot.py`, 
`python mnist/plot.py --task eig --loss nsgan`
