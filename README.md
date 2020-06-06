## Deep-generative-model-project
Final project of deep-generative-model in Spring 2020

This is the code using tensorflow for reproducing part experimental results in paper [A Closer Look at the Optimization Landscapes of Generative Adversarial Networks](https://arxiv.org/abs/1906.04848) and some other similar results. Two datasets: Mixture of Gaussian and MNIST are available. Four objective loss: NSGAN, LSGAN, WGAN, WGAN-GP are available.

We run this script under [TensorFlow](https://www.tensorflow.org) 2.0 and the [TensorLayer](https://github.com/tensorlayer/tensorlayer) 2.0+. 

For any questions regarding the code please contact lindachao@pku.edu.cn.

### Prepare Data 

- 1. Generate Mixture of Gaussian (MoG) dataset:

```python
python data_gen.py
```
- 2. Download MNIST dataset.

### Run
- To run the code for the Mixture of Gaussian experiment:
```python
python gaussian/train.py --epochs 100000 --en_dim 1 --save_dir results --loss wgan
```

-To run the code for the MNIST experiment:
```python
python mnist/train.py --epochs 50 --en_dim 5 --save_dir results --loss nsgan
```

### Visualization
The visualization of the results can be done with `plot.py`, including plotting eigenvalue of jacobian matrix, eigenvalue of hessian matrix of generator or discriminator loss function, and distribution or visualization of generator samples. 

```python
python mnist/plot.py --task eig --loss nsgan
```


### Reference
* [1] [A Closer Look at the Optimization Landscapes of Generative Adversarial Networks](https://arxiv.org/abs/1906.04848)

### Author
- [dachaolin](https://github.com/shayebuhui)

### Citation
If you find this project useful, I would be grateful if you cite the TensorLayer paper：

```
@article{tensorlayer2017,
author = {Dong, Hao and Supratak, Akara and Mai, Luo and Liu, Fangde and Oehmichen, Axel and Yu, Simiao and Guo, Yike},
journal = {ACM Multimedia},
title = {{TensorLayer: A Versatile Library for Efficient Deep Learning Development}},
url = {http://tensorlayer.org},
year = {2017}
}
```

### License

- For academic and non-commercial use only.
