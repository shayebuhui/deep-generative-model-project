import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Input, Dense, DeConv2d, Reshape, BatchNorm2d, Conv2d, Flatten, Dropout, MaxPool2d


def get_generator(shape, gf_dim=64, o_size=28, o_channel=1): # Dimension of gen filters in first conv layer. [64]
    image_size = o_size
    s4 = image_size // 4  ## DeConv twice to get original size image

    ni = Input(shape)
    nn = Reshape(shape=(-1, 1, 1, shape[1]))(ni)
    nn = DeConv2d(gf_dim, (s4, s4), (1, 1), padding='VALID')(nn)
    print(nn.shape)
    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu)(nn)
    nn = DeConv2d(gf_dim, (4, 4), (2, 2))(nn)
    print(nn.shape)
    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu)(nn)
    nn = DeConv2d(o_channel, (4, 4), (2, 2), act=tf.nn.tanh)(nn)
    print(nn.shape)
    return tl.models.Model(inputs=ni, outputs=nn)#, name='generator')


def get_discriminator(shape, df_dim=64): # Dimension of discrim filters in first conv layer. [64]

    ni = Input(shape)
    nn = Conv2d(df_dim, (4, 4), (2, 2), act=tf.nn.relu)(ni)
    nn = Conv2d(df_dim, (4, 4), (2, 2))(nn)  
    print(nn.shape)
    nn = BatchNorm2d(decay=0.9, act=tf.nn.relu)(nn)
    nn = Flatten()(nn)
    nn = Dense(n_units=1)(nn)
    print(nn.shape)
    return tl.models.Model(inputs=ni, outputs=nn)#, name='discriminator')

