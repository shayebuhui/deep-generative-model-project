import tensorlayer as tl
import tensorflow as tf
from model import get_generator, get_discriminator
import numpy as np
import scipy

## Gradient penalty term for WGAN-GP
def gradient_penalty(x, z, model):
    assert x.shape == z.shape
    alpha = tf.random.uniform(shape=[x.shape[0]] + [1] * (len(x.shape) - 1), minval=0., maxval=1.)
    interpolates = (1 - alpha) * z + alpha * x
    with tf.GradientTape(persistent=True) as grad_tape:
        grad_tape.watch(interpolates)
        logits = model(interpolates)
    gradients = grad_tape.gradient(logits, interpolates)

    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    
    del grad_tape
    return gradient_penalty

## Get matrix form of jacodian matrix
def hessian_matrix(grad_grads, grads):
    hessian = []
    for hs, g in zip(grad_grads, grads):
        g_shape = tf.squeeze(g).shape
        if len(g_shape) == 0:
            x = 1
        else:
            x = g_shape[0]
        hess_rows = []    
        for h in hs:
            sq_h = tf.squeeze(h)
            if len(h.shape) != 2:
                h = tf.reshape(h, [x, -1])   
            hess_rows.append(h)
        hess_rows = tf.concat(hess_rows, axis=1)
        hessian.append(hess_rows)
    return tf.concat(hessian, axis=0).numpy()

## Get jacobian matrix and game gradient vector
def get_grad_hessian(D, G, trainset, loss_type, use_hessian=True):
    with tf.GradientTape(persistent=True) as hess_tape:
        with tf.GradientTape(persistent=True) as grad_tape:
            z = np.random.normal(loc=0.0, scale=1.0, size=[10000, 1]).astype(np.float32)
            d_logits = D(G(z))
            d2_logits = D(trainset)

            if loss_type == 'lsgan':
                d_loss_real = tl.cost.mean_squared_error(tf.sigmoid(d2_logits), tf.ones_like(d2_logits), is_mean=True)
                d_loss_fake = tl.cost.mean_squared_error(tf.sigmoid(d_logits), tf.zeros_like(d_logits), is_mean=True)
            if loss_type == 'nsgan':
                d_loss_real = tl.cost.sigmoid_cross_entropy(d2_logits, tf.ones_like(d2_logits), name='dreal')
                d_loss_fake = tl.cost.sigmoid_cross_entropy(d_logits, tf.zeros_like(d_logits), name='dfake')
            if loss_type in ['wgan', 'wgan-gp']:
                d_loss_real = -tf.reduce_mean(d2_logits)
                d_loss_fake = tf.reduce_mean(d_logits) 

            # combined loss for updating discriminator
            d_loss = d_loss_real + d_loss_fake
            if loss_type == 'wgan-gp':
                penalty = 0.001 * gradient_penalty(trainset, G(z), D)
                d_loss += penalty
            # generator: try to fool discriminator to output 1
            if loss_type == 'lsgan':
                g_loss = tl.cost.mean_squared_error(tf.sigmoid(d_logits), tf.ones_like(d_logits), is_mean=True)
            if loss_type == 'nsgan':    
                g_loss = tl.cost.sigmoid_cross_entropy(d_logits, tf.ones_like(d_logits), name='gfake')   
            if loss_type in ['wgan', 'wgan-gp']:
                g_loss = -tf.reduce_mean(d_logits) 
        
        grads_D = grad_tape.gradient(d_loss, D.trainable_weights, 
                          unconnected_gradients=tf.UnconnectedGradients.ZERO)
        grads_G = grad_tape.gradient(g_loss, G.trainable_weights, 
                          unconnected_gradients=tf.UnconnectedGradients.ZERO)
    if use_hessian:
        hessian_DD = [hess_tape.jacobian(g, D.trainable_weights, 
                 unconnected_gradients=tf.UnconnectedGradients.ZERO) for g in grads_D]
        hessian_DG = [hess_tape.jacobian(g, G.trainable_weights, 
                 unconnected_gradients=tf.UnconnectedGradients.ZERO) for g in grads_D]
        hessian_GD = [hess_tape.jacobian(g, D.trainable_weights, 
                 unconnected_gradients=tf.UnconnectedGradients.ZERO) for g in grads_G]
        hessian_GG = [hess_tape.jacobian(g, G.trainable_weights, 
                 unconnected_gradients=tf.UnconnectedGradients.ZERO) for g in grads_G]
        hessian_DD = hessian_matrix(hessian_DD, grads_D)
        hessian_DG = hessian_matrix(hessian_DG, grads_D)
        hessian_GD = hessian_matrix(hessian_GD, grads_G)
        hessian_GG = hessian_matrix(hessian_GG, grads_G)
        hessian = np.block([[hessian_DD, hessian_DG], [hessian_GD, hessian_GG]])    
    grads_D = tf.concat([tf.reshape(grad, [-1,]) for grad in grads_D], axis=0).numpy()
    grads_G = tf.concat([tf.reshape(grad, [-1,]) for grad in grads_G], axis=0).numpy()
    if use_hessian:
        return grads_D, grads_G, hessian_DD, hessian_GG, hessian
    else:
        return grads_D, grads_G

def interpolate(model, model1, model2, ratio):
    for weight, weight1, weight2 in zip(model.all_weights, model1.all_weights, model2.all_weights):
        weight.assign((1 - ratio) * weight1 + ratio * weight2)

## Get path norm and angle results        
def path_angle(D, G, D1, D2, G1, G2, trainset, loss_type):
    diff_D = tf.concat([tf.reshape(weights1-weights2, [-1, ]) for weights1, weights2 in zip(D1.all_weights, D2.all_weights)], axis=0).numpy()
    diff_G = tf.concat([tf.reshape(weights1-weights2, [-1, ]) for weights1, weights2 in zip(G1.all_weights, G2.all_weights)], axis=0).numpy()
    diff_D_norm = np.linalg.norm(diff_D)
    diff_G_norm = np.linalg.norm(diff_G)
    v = np.concatenate([diff_D, diff_G], axis=0)
    diff_norm = np.linalg.norm(v)
    angle_list = []
    norm_g = []
    for ratio in np.linspace(-1, 2, 31):
        interpolate(D, D1, D2, ratio)
        interpolate(G, G1, G2, ratio)
        grads_D, grads_G = get_grad_hessian(D, G, trainset, loss_type, use_hessian=False)
        grads = np.concatenate([grads_D, grads_G], axis=0)
        if np.linalg.norm(grads) == 0:
            angle_list.append(1)
        else:    
            angle_list.append(1 - scipy.spatial.distance.cosine(grads, v))
        norm_g.append(np.linalg.norm(grads))

    return angle_list, norm_g