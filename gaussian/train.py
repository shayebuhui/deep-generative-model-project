import os, time, multiprocessing
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from glob import glob
import argparse
from model import get_generator, get_discriminator
from utils import gradient_penalty

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser(description='Gaussian GAN Training')

# Dataset
parser.add_argument('--bs', default=10000, type=int, help='Batch size.')

# Optimization options
parser.add_argument('--epochs', default=100000, type=int, help='Training epochs.')
parser.add_argument('--lr', default=1e-4, type=str, help='Learning rate.')
parser.add_argument('--beta1', default=0.5, type=str, help='Decay rate in Adam.')
parser.add_argument('--loss', default='nsgan', type=str, help='Loss type in GAN.')
parser.add_argument('--clip', type=float, default=0.01, help='weight clip range (wgan)')
parser.add_argument('--gp_lambda', type=float, default=0.001, help='weight for gradient penalty (wgan_gp)')

# Architecture
parser.add_argument('--en_dim', default=1, type=int, help='Encoder dimension.')

# I/O args
parser.add_argument('--save_freq', default=1000, type=int, help='The interval of saveing checkpoints.')
parser.add_argument('--save_dir', default='./results' , type=str, help='Save directory.')

args = parser.parse_args()
args.save_dir += '_' + args.loss
tl.files.exists_or_mkdir(args.save_dir) # save model

def get_gaussian(batch_size):
    trainset = np.load('./data.npy')
    length = len(trainset)
    train_ds = tf.data.Dataset.from_tensor_slices(trainset)
    ds = train_ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=2)
    return ds, length


def train():
    images, len_instance = get_gaussian(args.bs)

    G = get_generator([None, args.en_dim], h=50)
    D = get_discriminator([None, 1], h=50)

    G.save_weights('{}/G_init.npz'.format(args.save_dir), format='npz')
    D.save_weights('{}/D_init.npz'.format(args.save_dir), format='npz')
        
    G.train()
    D.train()

    d_optimizer = tf.optimizers.Adam(0.0005, beta_1=args.beta1)
    g_optimizer = tf.optimizers.Adam(0.0005, beta_1=args.beta1)

    n_step_epoch = int(len_instance // args.bs)
    
    for epoch in range(args.epochs):
        for step, batch_images in enumerate(images):
            if batch_images.shape[0] != args.bs:  # if the remaining data in this epoch < batch_size
                break
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                z = np.random.normal(loc=0.0, scale=1.0, size=[args.bs, args.en_dim]).astype(np.float32)
                d_logits = D(G(z))
                d2_logits = D(batch_images)
                
                if args.loss == 'lsgan':
                    d_loss_real = tl.cost.mean_squared_error(tf.sigmoid(d2_logits), tf.ones_like(d2_logits), is_mean=True)
                    d_loss_fake = tl.cost.mean_squared_error(tf.sigmoid(d_logits), tf.zeros_like(d_logits), is_mean=True)
                if args.loss == 'nsgan':
                    d_loss_real = tl.cost.sigmoid_cross_entropy(d2_logits, tf.ones_like(d2_logits), name='dreal')
                    d_loss_fake = tl.cost.sigmoid_cross_entropy(d_logits, tf.zeros_like(d_logits), name='dfake')
                if args.loss == 'wgan' or args.loss == 'wgan-gp':
                    d_loss_real = -tf.reduce_mean(d2_logits)
                    d_loss_fake = tf.reduce_mean(d_logits) 

                # combined loss for updating discriminator
                d_loss = d_loss_real + d_loss_fake
                if args.loss == 'wgan-gp':
                    penalty = args.gp_lambda * gradient_penalty(batch_images, G(z), D)
                    d_loss += penalty
                if args.loss == 'lsgan':
                    g_loss = tl.cost.mean_squared_error(tf.sigmoid(d_logits), tf.ones_like(d_logits), is_mean=True)
                if args.loss == 'nsgan':
                    g_loss = tl.cost.sigmoid_cross_entropy(d_logits, tf.ones_like(d_logits), name='gfake')   
                if args.loss == 'wgan' or args.loss == 'wgan-gp':
                    g_loss = -tf.reduce_mean(d_logits) 

            grad = tape.gradient(g_loss, G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights)) 
            
            grad = tape.gradient(d_loss, D.trainable_weights)
            d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
            
            del tape
            
            if args.loss == 'wgan':
                for weights in D.trainable_weights:
                    weights.assign(tf.clip_by_value(weights, -args.clip, args.clip))
                    
            print("Epoch: [{}/{}] [{}/{}] took: {:.3f}, d_loss: {:.5f}, g_loss: {:.5f}".format(
                epoch, args.epochs, step, n_step_epoch, time.time() - step_time, d_loss, g_loss.numpy()))

        if np.mod(epoch+1, args.save_freq) == 0:
            G.save_weights('{}/G_{:d}.npz'.format(args.save_dir, epoch+1), format='npz')
            D.save_weights('{}/D_{:d}.npz'.format(args.save_dir, epoch+1), format='npz')
            G.eval()
            result = G(z)
            np.save('./{}/result_{:d}'.format(args.save_dir, epoch+1), result.numpy())


if __name__ == '__main__':
    train()