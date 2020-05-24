import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import Input, Dense, DeConv2d, Reshape, BatchNorm2d, Conv2d, Flatten, Dropout, MaxPool2d


def get_generator(shape, h=100):
    ni = Input(shape)
    nn = Dense(n_units=h, act=tf.nn.relu)(ni)
    nn = Dense(n_units=1, b_init=None)(nn)
    return tl.models.Model(inputs=ni, outputs=nn)#, name='generator')


def get_discriminator(shape, h=100):
    ni = Input(shape)
    nn = Dense(n_units=h, act=tf.nn.relu)(ni)
    nn = Dense(n_units=1, b_init=None)(nn)
    return tl.models.Model(inputs=ni, outputs=nn)#, name='discriminator')
