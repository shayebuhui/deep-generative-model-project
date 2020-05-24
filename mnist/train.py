import os, time, multiprocessing
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from glob import glob
import argparse
from model import get_generator, get_discriminator
import skimage.transform
from utils import gradient_penalty

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser(description='MNIST GAN Training')

# Dataset
parser.add_argument('--bs', default=100, type=int, help='Batch size.')

# Optimization options
parser.add_argument('--epochs', default=50, type=int, help='Training epochs.')
parser.add_argument('--lr', default=1e-4, type=str, help='Learning rate.')
parser.add_argument('--beta1', default=0.5, type=str, help='Decay rate in Adam.')
parser.add_argument('--loss', default='nsgan', type=str, help='Loss type in GAN.')
parser.add_argument('--clip', type=float, default=0.1, help='weight clip range (wgan)')
parser.add_argument('--n_critic', type=int, default=1, help='number of critic updates per generator update (wgan/wgan_gp)')
parser.add_argument('--gp_lambda', type=float, default=50, help='weight for gradient penalty (wgan_gp)')

# Architecture
parser.add_argument('--en_dim', default=5, type=int, help='Encoder dimension.')
parser.add_argument('--output_size', default=28, type=int)

# I/O args
parser.add_argument('--sample_size', default=100, type=int)
parser.add_argument('--channels', default=1, type=int, help='Output channels')
parser.add_argument('--save_freq', default=1, type=int, help='The interval of saveing checkpoints.')
parser.add_argument('--save_dir', default='results' , type=str, help='Save directory.')
parser.add_argument('--sample_dir', default='samples' , type=str, help='The number of sample images.')

args = parser.parse_args()

num_tiles = int(np.sqrt(args.sample_size))

args.save_dir = 'results_' + args.loss + '/results'
args.sample_dir = 'results_' + args.loss + '/' + args.sample_dir
tl.files.exists_or_mkdir(args.save_dir) # save model
tl.files.exists_or_mkdir(args.sample_dir) # save generated image

def get_mnist(batch_size):
    trainset, y_train, valset, y_val, testset, y_test = tl.files.load_mnist_dataset(shape=(-1, 784))
    length = len(trainset)
#     img = skimage.transform.resize(img.reshape([28, 28, 1]), (16, 16))
        
    def transform():
        for img in trainset:
            yield (img.reshape([28, 28, 1]) - 0.5) / 0.5  ## Normalize mnist image

    train_ds = tf.data.Dataset.from_generator(transform, output_types=tf.float32)
    ds = train_ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=2)
    return ds, length

def train():
    images, len_instance = get_mnist(args.bs)
    G = get_generator([None, args.en_dim], gf_dim=8, o_size=args.output_size, o_channel=args.channels)
    D = get_discriminator([None, args.output_size, args.output_size, args.channels], df_dim=8)
    G.save_weights('{}/G_init.npz'.format(args.save_dir), format='npz')
    D.save_weights('{}/D_init.npz'.format(args.save_dir), format='npz')
    G.train()
    D.train()

    d_optimizer = tf.optimizers.Adam(args.lr, beta_1=args.beta1)
    g_optimizer = tf.optimizers.Adam(args.lr, beta_1=args.beta1)

    n_step_epoch = int(len_instance // args.bs)
    
    for epoch in range(args.epochs):
        for step, batch_images in enumerate(images):
            if batch_images.shape[0] != args.bs:  # if the remaining data in this epoch < batch_size
                break
#             batch_images = tf.image.resize(batch_images, [16, 16])
#             print(batch_images.shape)
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
#                     print(penalty)
                    d_loss += penalty
                if args.loss == 'lsgan':
                    g_loss = tl.cost.mean_squared_error(tf.sigmoid(d_logits), tf.ones_like(d_logits), is_mean=True)
                if args.loss == 'nsgan':    
                    g_loss = tl.cost.sigmoid_cross_entropy(d_logits, tf.ones_like(d_logits), name='gfake')   
                if args.loss == 'wgan' or args.loss == 'wgan-gp':
                    g_loss = -tf.reduce_mean(d_logits)     

            grad = tape.gradient(d_loss, D.trainable_weights)
            d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
            if args.loss in ['wgan', 'wgan-gp'] and step % args.n_critic == 0:
                grad = tape.gradient(g_loss, G.trainable_weights)
                g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            
            del tape
            
            if args.loss == 'wgan':
                for weights in D.trainable_weights:
                    weights.assign(tf.clip_by_value(weights, -args.clip, args.clip))

            print("Epoch: [{}/{}] [{}/{}] took: {:.3f}, d_loss: {:.5f}, g_loss: {:.5f}".format(
                epoch, args.epochs, step, n_step_epoch, time.time() - step_time, d_loss, g_loss))

        if np.mod(epoch, args.save_freq) == 0:
            G.save_weights('{}/G_{}.npz'.format(args.save_dir, epoch), format='npz')
            D.save_weights('{}/D_{}.npz'.format(args.save_dir, epoch), format='npz')
            G.eval()
            result = G(z)
            G.train()
            tl.visualize.save_images(result.numpy(), [num_tiles, num_tiles], '{}/train_{:02d}.png'.format(args.sample_dir, epoch))


if __name__ == '__main__':
    train()