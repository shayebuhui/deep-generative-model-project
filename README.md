# Deep-generative-model-project
Final project of deep-generative-model in Spring 2020

This is the code for reproducing the experimental results in paper [A Closer Look at the Optimization Landscapes of Generative Adversarial Networks](https://arxiv.org/abs/1906.04848), Hugo Berard, Gauthier Gidel,  Amjad Almahairi, Pascal Vincent, Simon Lacoste-Julien, 2019.


For any questions regarding the code please contact Hugo Berard (berard.hugo@gmail.com).

## Running the Code

The code for computing the eigenvalues and the path-angle is in `plot.py`.

To run the code for the Mixture of Gaussian experiment:
`python gaussian/train.py --save_dir `

To run the code for the MNIST experiment:
`python mnist/train.py`

The visualization of the results can be done with `plot.py`
