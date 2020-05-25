import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorlayer as tl
import tensorflow as tf
import numpy as np
from utils import *
import scipy
from model import get_generator, get_discriminator
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rcParams
from matplotlib.ticker import MaxNLocator
from plotlib import _plot_avg_cosine_similarity, _plot_eigenvalues
import pickle

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
# cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
# print(gpus, cpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
def get_mnist(batch_size):
    trainset, y_train, valset, y_val, testset, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))
    length = len(trainset)        
    def transform():
        for img in trainset:
            yield (img.reshape([28, 28, 1]) - 0.5) / 0.5  ## Normalize mnist image

    train_ds = tf.data.Dataset.from_generator(transform, output_types=tf.float32)
    ds = train_ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=2)
    return ds, length

def generate_interpolate(path, epoch1, epoch2, en_dim=5):
    z = np.random.normal(loc=0.0, scale=1.0, size=[30, en_dim]).astype(np.float32)
    G1 = get_generator([None, en_dim], gf_dim=8, o_size=28, o_channel=1)
    G1.load_weights(path + '/G_' + str(epoch1) + '.npz')
    G1.train()
    G2 = get_generator([None, en_dim], gf_dim=8, o_size=28, o_channel=1)
    G2.load_weights(path + '/G_' + str(epoch2) + '.npz')
    G2.train()
    result1 = G1(z).numpy()
    result2 = G2(z).numpy()

    res = []
    for ratio in np.linspace(0, 1, 9):
        res.append((1 - ratio) * result1 + ratio * result2)

    tl.visualize.save_images(np.concatenate(res, axis=0), [9, 30], 'figure_%s/inter_%d_%d.png' % (path, epoch1, epoch2))

## Plot Path-norm and Path-angle     
def plot_avg_cosine_similarity(path, epoch1, epoch2, loss_type, en_dim=5):
    trainset, len_instance = get_mnist(10000)
    D1 = get_discriminator([None, 28, 28, 1], df_dim=8)
    G1 = get_generator([None, en_dim], gf_dim=8, o_size=28, o_channel=1)
    if epoch1 == 'init':
        D1.load_weights(path + '/D_init.npz')
        G1.load_weights(path + '/G_init.npz')
    else:
        D1.load_weights(path + '/D_' + str(epoch1) + '.npz')
        G1.load_weights(path + '/G_' + str(epoch1) + '.npz')
        
    D2 = get_discriminator([None, 28, 28, 1], df_dim=8)
    D2.load_weights(path + '/D_' + str(epoch2) + '.npz')
    G2 = get_generator([None, en_dim], gf_dim=8, o_size=28, o_channel=1)
    G2.load_weights(path + '/G_' + str(epoch2) + '.npz')
    
    D = get_discriminator([None, 28, 28, 1], df_dim=8)
    G = get_generator([None, en_dim], gf_dim=8, o_size=28, o_channel=1)
    D1.train()
    G1.train()
    D2.train()
    G2.train()
    D.train()
    G.train()
    angle_list, norm_g = path_angle(D, G, D1, D2, G1, G2, trainset, loss_type)
    angle_list = np.array(angle_list).reshape([1, -1])
    norm_g = np.array(norm_g).reshape([1, -1])
    os.makedirs('./figure_%s' % path, exist_ok=True)
    np.save('./figure_%s/angle_%s_%s.npy' % (path, epoch1, epoch2), np.array(angle_list))
    np.save('./figure_%s/norm_g_%s_%s.npy' % (path, epoch1, epoch2), np.array(norm_g))
    _plot_avg_cosine_similarity(np.linspace(-1, 2, 31), angle_list, norm_g, 'figure_' + path, 'angle_norm_%s_%s' % (epoch1, epoch2))

## Plot eigenvalue of jacobian matrix and hessian matrix in each loss 
def plot_eigenvalues(path, epoch, loss_type, en_dim=5):
    trainset, len_instance = get_mnist(10000)
    D = get_discriminator([None, 28, 28, 1], df_dim=8)
    D.load_weights(path + '/D_init.npz')
    G = get_generator([None, en_dim], gf_dim=8, o_size=28, o_channel=1)
    G.load_weights(path + '/G_init.npz')
    D.train()
    G.train()
    _, _, hessian_DD, hessian_GG, hessian = get_grad_hessian(D, G, trainset, loss_type, use_hessian=True)
    u, v = scipy.sparse.linalg.eigs(hessian, k=20)
    u_G, v_G = scipy.sparse.linalg.eigs(hessian_GG, k=20)
    u_D, v_D = scipy.sparse.linalg.eigs(hessian_DD, k=20)
    dct_init = {'game_eigs': u, 'dis_eigs': u_D.real, 'gen_eigs': u_G.real}
    
    D = get_discriminator([None, 28, 28, 1], df_dim=8)
    D.load_weights(path + '/D_' + str(epoch) + '.npz')
    G = get_generator([None, en_dim], gf_dim=8, o_size=28, o_channel=1)
    G.load_weights(path + '/G_' + str(epoch) + '.npz')
    D.train()
    G.train()
    _, _, hessian_DD, hessian_GG, hessian = get_grad_hessian(D, G, trainset, loss_type, use_hessian=True)
    u, v = scipy.sparse.linalg.eigs(hessian, k=20)
    u_G, v_G = scipy.sparse.linalg.eigs(hessian_GG, k=20)
    u_D, v_D = scipy.sparse.linalg.eigs(hessian_DD, k=20)
    dct_end = {'game_eigs': u, 'dis_eigs': u_D.real, 'gen_eigs': u_G.real}
    
    pickle.dump(dct_init, open('./figure_%s/eig_init_%s' % (path, epoch), 'wb'))
    pickle.dump(dct_end, open('./figure_%s/eig_end_%s' % (path, epoch), 'wb'))
    
    _plot_eigenvalues(dct_init, dct_end, ['init', 'end'], 'figure_' + path, 'f-' + str(epoch))
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Gaussian GAN Plot.')
    parser.add_argument('--epoch', default=40, type=int, help='Epoch for figure.')
    parser.add_argument('--loss', default='wgan-gp', type=str, help='GAN loss.')
    parser.add_argument('--task', default='dis', type=str, help='Plot task.', choices=['gen', 'eig', 'path'])
    parser.add_argument('--epoch1', default=40, type=str, help='Epoch for figure.')
    parser.add_argument('--epoch2', default=49, type=str, help='Epoch for figure.')
    
    import time
    time_start = time.time()

    args = parser.parse_args()
    path = 'results_' + args.loss + '/results'
    loss_type = args.loss

    if args.task == 'eig': 
        plot_eigenvalues(path, args.epoch, loss_type, en_dim=5)
    if args.task == 'path':    
        plot_avg_cosine_similarity(path, 'init', args.epoch1, loss_type, en_dim=5)
        plot_avg_cosine_similarity(path, 'init', args.epoch2, loss_type, en_dim=5)
        plot_avg_cosine_similarity(path, args.epoch1, args.epoch2, loss_type, en_dim=5)
    if args.task == 'gen':
        generate_interpolate(path, args.epoch1, args.epoch2, en_dim=5)

    time_end = time.time()
    print('time cost',time_end-time_start,'s')