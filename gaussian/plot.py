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
    
## Generator Distribution
def hist_com(path, epoch):
    G = get_discriminator([None, 1], h=50)
    G.load_weights(path + '/G_' + str(epoch) + '.npz')
    G.eval()
    z = np.random.normal(loc=0.0, scale=1.0, size=[10000, 1]).astype(np.float32)
    plt.hist(G(z).numpy(), 100)
    trainset = np.load('./data.npy')
    plt.hist(trainset, 100)
    os.makedirs('figure_%s' % path, exist_ok=True)
    plt.savefig('figure_%s/dis_%d.jpg' % (path, epoch), format='jpg', bbox_inches="tight")

## Generator distribution between two training epochs
def hist_com_inter(path, epoch1, epoch2):
    G1 = get_discriminator([None, 1], h=50)
    G1.load_weights(path + '/G_' + str(epoch1) + '.npz')
    G1.eval()
    G2 = get_discriminator([None, 1], h=50)
    G2.load_weights(path + '/G_' + str(epoch2) + '.npz')
    G2.eval()
    z = np.random.normal(loc=0.0, scale=1.0, size=[10000, 1]).astype(np.float32)
    fig, axlist = plt.subplots(3, 3, figsize=(10, 10))
    ind = 0
    for ratio in np.linspace(0, 1, 9):
        ax = axlist[ind // 3][ind % 3]
        ax.hist((1 - ratio) * G1(z).numpy() + ratio * G2(z).numpy(), 100)
        trainset = np.load('./data.npy')
        ax.hist(trainset, 100)
        ax.set_title(r'ratio: %s' % ratio)
        ind += 1
    os.makedirs('figure_%s' % path, exist_ok=True)
    plt.tight_layout()
    plt.savefig('figure_%s/dis_%d_%d.jpg' % (path, epoch1, epoch2), format='jpg', bbox_inches="tight")

## Plot Path-norm and Path-angle    
def plot_avg_cosine_similarity(path, epoch1, epoch2, loss_type):
    trainset = np.load('./data.npy')
    D1 = get_discriminator([None, 1], h=50)
    if epoch1 == 'init':
        D1.load_weights(path + '/D_init.npz')
    else:
        D1.load_weights(path + '/D_' + str(epoch1) + '.npz')
    
    D2 = get_discriminator([None, 1], h=50)
    D2.load_weights(path + '/D_' + str(epoch2) + '.npz')
    
    G1 = get_generator([None, 1], h=50)
    if epoch1 == 'init':
        G1.load_weights(path + '/G_init.npz')
    else:
        G1.load_weights(path + '/G_' + str(epoch1) + '.npz')
    
    G2 = get_generator([None, 1], h=50)
    G2.load_weights(path + '/G_' + str(epoch2) + '.npz')
    
    D = get_discriminator([None, 1], h=50)
    G = get_generator([None, 1], h=50)
    D1.eval()
    G1.eval()
    D2.eval()
    G2.eval()
    D.eval()
    G.eval()
    angle_list, norm_g = path_angle(D, G, D1, D2, G1, G2, trainset, loss_type)
    angle_list = np.array(angle_list).reshape([1, -1])
    norm_g = np.array(norm_g).reshape([1, -1])
    os.makedirs('./figure_%s' % path, exist_ok=True)
    np.save('./figure_%s/angle_%s_%s.npy' % (path, epoch1, epoch2), np.array(angle_list))
    np.save('./figure_%s/norm_g_%s_%s.npy' % (path, epoch1, epoch2), np.array(norm_g))
    _plot_avg_cosine_similarity(np.linspace(-1, 2, 31), angle_list, norm_g, 'figure_' + path, 'angle_norm_%s_%s' % (epoch1, epoch2))

## Plot eigenvalue of jacobian matrix and hessian matrix in each loss   
def plot_eigenvalues(path, epoch, loss_type):
    trainset = np.load('./data.npy')
    D = get_discriminator([None, 1], h=50)
    D.load_weights(path + '/D_init.npz')
    G = get_generator([None, 1], h=50)
    G.load_weights(path + '/G_init.npz')
    D.eval()
    G.eval()
    _, _, hessian_DD, hessian_GG, hessian = get_grad_hessian(D, G, trainset, loss_type, use_hessian=True)
    u, v = scipy.sparse.linalg.eigs(hessian, k=20)
    u_G, v_G = scipy.sparse.linalg.eigs(hessian_GG, k=20)
    u_D, v_D = scipy.sparse.linalg.eigs(hessian_DD, k=20)
    dct_init = {'game_eigs': u, 'dis_eigs': u_D.real, 'gen_eigs': u_G.real}
    
    D = get_discriminator([None, 1], h=50)
    D.load_weights(path + '/D_' + str(epoch) + '.npz')
    G = get_generator([None, 1], h=50)
    G.load_weights(path + '/G_' + str(epoch) + '.npz')
    D.eval()
    G.eval()
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
    parser.add_argument('--epoch', default=20000, type=int, help='Epoch for figure.')
    parser.add_argument('--loss', default='wgan-gp', type=str, help='GAN loss.')
    parser.add_argument('--task', default='dis', type=str, help='Plot task.', choices=['dis', 'eig', 'path'])
    parser.add_argument('--epoch1', default=20000, type=str, help='Epoch for figure.')
    parser.add_argument('--epoch2', default=30000, type=str, help='Epoch for figure.')
    
    args = parser.parse_args()
    path = 'results_' + args.loss
    loss_type = args.loss
    epoch = args.epoch
    
    if args.task == 'dis':
        hist_com(path, args.epoch1)
        hist_com_inter(path, args.epoch1, args.epoch2)
    if args.task == 'eig':   
        plot_eigenvalues(path, epoch, loss_type)
    if args.task == 'path':     
        plot_avg_cosine_similarity(path, 'init', args.epoch1, loss_type)
        plot_avg_cosine_similarity(path, 'init', args.epoch2, loss_type)
        plot_avg_cosine_similarity(path, args.epoch1, args.epoch2, loss_type)   
