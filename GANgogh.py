import os, sys
sys.path.append(os.getcwd())

from random import randint

import time
import functools
import math

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.wikiartGenre
import tflib.ops.layernorm
import tflib.plot

MODE = 'acwgan'  # dcgan, wgan, wgan-gp, lsgan
DIM = 64  # Model dimensionality
CRITIC_ITERS = 5  # How many iterations to train the critic for
N_GPUS = 1  # Number of GPUs
LAMBDA = 10  # Gradient penalty lambda hyperparameter
OUTPUT_DIM = 64 * 64 * 3  # Number of pixels in each iamge


BATCH_SIZE = 84  # Batch size. Must be a multiple of CLASSES and N_GPUS
CLASSES = 14  # Number of classes, for genres probably 14



lib.print_model_settings(locals().copy())


def GeneratorAndDiscriminator():
    return kACGANGenerator, kACGANDiscriminator


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
    return LeakyReLU(output)

def Batchnorm(name, axes, inputs):
    
    if ('Discriminator' in name) and (MODE == 'wgan-gp' or MODE == 'acwgan'):
        if axes != [0,2,3]:
            raise Exception('Layernorm over non-standard axes is unsupported')
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
    else:
        return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=True)

def pixcnn_gated_nonlinearity(name, output_dim, a, b, c=None, d=None):
    if c is not None and d is not None:
        a = a + c
        b = b + d
        
    result = tf.sigmoid(a) * tf.tanh(b)
    return result

def SubpixelConv2D(*args, **kwargs):
    kwargs['output_dim'] = 4*kwargs['output_dim']
    output = lib.ops.conv2d.Conv2D(*args, **kwargs)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D, stride=2)
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim//2)
        conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim//2, output_dim=output_dim//2, stride=2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim//2, output_dim=output_dim)
    elif resample=='up':
        conv_shortcut = SubpixelConv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim//2)
        conv_1b       = functools.partial(lib.ops.deconv2d.Deconv2D, input_dim=input_dim//2, output_dim=output_dim//2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim//2, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim//2)
        conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim//2,  output_dim=output_dim//2)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim//2, output_dim=output_dim)

    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                 he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = tf.nn.relu(output)
    output = conv_1(name+'.Conv1', filter_size=1, inputs=output, he_init=he_init, weightnorm=False)
    output = tf.nn.relu(output)
    output = conv_1b(name+'.Conv1B', filter_size=filter_size, inputs=output, he_init=he_init, weightnorm=False)
    output = tf.nn.relu(output)
    output = conv_2(name+'.Conv2', filter_size=1, inputs=output, he_init=he_init, weightnorm=False, biases=False)
    output = Batchnorm(name+'.BN', [0,2,3], output)

    return shortcut + (0.3*output)

# ! Generators

def kACGANGenerator(n_samples, numClasses, labels, noise=None, dim=DIM, bn=True, nonlinearity=tf.nn.relu, condition=None):
    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)
    lib.ops.linear.set_weights_stdev(0.02)
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    labels = tf.cast(labels, tf.float32)        
    noise = tf.concat([noise, labels], 1)

    output = lib.ops.linear.Linear('Generator.Input', 128+numClasses, 8*4*4*dim*2, noise) #probs need to recalculate dimensions
    output = tf.reshape(output, [-1, 8*dim*2, 4, 4])
    if bn:
        output = Batchnorm('Generator.BN1', [0,2,3], output)
    condition = lib.ops.linear.Linear('Generator.cond1', numClasses, 8*4*4*dim*2, labels,biases=False)
    condition = tf.reshape(condition, [-1, 8*dim*2, 4, 4])
    output = pixcnn_gated_nonlinearity('Generator.nl1', 8*dim, output[:,::2], output[:,1::2], condition[:,::2], condition[:,1::2])


    output = lib.ops.deconv2d.Deconv2D('Generator.2', 8*dim, 4*dim*2, 5, output)
    if bn:
        output = Batchnorm('Generator.BN2', [0,2,3], output)
    condition = lib.ops.linear.Linear('Generator.cond2', numClasses, 4*8*8*dim*2, labels)
    condition = tf.reshape(condition, [-1, 4*dim*2, 8, 8])
    output = pixcnn_gated_nonlinearity('Generator.nl2', 4*dim,output[:,::2], output[:,1::2], condition[:,::2], condition[:,1::2])
    
    output = lib.ops.deconv2d.Deconv2D('Generator.3', 4*dim, 2*dim*2, 5, output)
    if bn:
        output = Batchnorm('Generator.BN3', [0,2,3], output)
    condition = lib.ops.linear.Linear('Generator.cond3', numClasses, 2*16*16*dim*2, labels)
    condition = tf.reshape(condition, [-1, 2*dim*2, 16, 16])
    output = pixcnn_gated_nonlinearity('Generator.nl3', 2*dim,output[:,::2], output[:,1::2], condition[:,::2], condition[:,1::2])
    
    output = lib.ops.deconv2d.Deconv2D('Generator.4', 2*dim, dim*2, 5, output)
    if bn:
        output = Batchnorm('Generator.BN4', [0,2,3], output)
    condition = lib.ops.linear.Linear('Generator.cond4', numClasses, 32*32*dim*2, labels)
    condition = tf.reshape(condition, [-1, dim*2, 32, 32])
    output = pixcnn_gated_nonlinearity('Generator.nl4', dim, output[:,::2], output[:,1::2], condition[:,::2], condition[:,1::2])

    output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)

    output = tf.tanh(output)
    
    lib.ops.conv2d.unset_weights_stdev()
    lib.ops.deconv2d.unset_weights_stdev()
    lib.ops.linear.unset_weights_stdev()

    return tf.reshape(output, [-1, OUTPUT_DIM]), labels

def kACGANDiscriminator(inputs, numClasses, dim=DIM, bn=True, nonlinearity=LeakyReLU):
    output = tf.reshape(inputs, [-1, 3, 64, 64])

    lib.ops.conv2d.set_weights_stdev(0.02)
    lib.ops.deconv2d.set_weights_stdev(0.02)
    lib.ops.linear.set_weights_stdev(0.02)
    
    output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, dim, 5, output, stride=2)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', dim, 2*dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = nonlinearity(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*dim, 4*dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = nonlinearity(output)

    
    output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*dim, 8*dim, 5, output, stride=2)
    if bn:
        output = Batchnorm('Discriminator.BN4', [0,2,3], output)
    output = nonlinearity(output)
    finalLayer = tf.reshape(output, [-1, 4*4*8*dim])

    sourceOutput = lib.ops.linear.Linear('Discriminator.sourceOutput', 4*4*8*dim, 1, finalLayer)
    
    classOutput = lib.ops.linear.Linear('Discriminator.classOutput', 4*4*8*dim, numClasses, finalLayer)

    lib.ops.conv2d.unset_weights_stdev()
    lib.ops.deconv2d.unset_weights_stdev()
    lib.ops.linear.unset_weights_stdev()



    return (tf.reshape(sourceOutput, [-1]), tf.reshape(classOutput, [-1, numClasses]))

                                
def genRandomLabels(n_samples, numClasses,condition=None):
    labels = np.zeros([BATCH_SIZE,CLASSES], dtype=np.float32)
    for i in range(n_samples):
        if condition is not None:
            labelNum = condition
        else:
            labelNum = randint(0, numClasses-1)
        labels[i, labelNum] = 1
    return labels
