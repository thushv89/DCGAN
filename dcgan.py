__author__ = 'Thushan Ganegedara'

import tensorflow as tf
import numpy as np
from math import ceil,floor
import load_data
import scipy.misc
import scipy
import os
import logging
import sys,getopt

epsilon = 1e-8
bn_decay = 0.5
loss_epsilon = 1e-8

dataset_type = 'wikiface'
# hyperparameters
if dataset_type=='cifar-10':
    image_size = 32
    num_labels = 10
    num_channels = 3 # rgb
elif dataset_type=='imagenet':
    image_size = 224
    num_labels = 100
    num_channels = 3 # grayscale
elif dataset_type=='wikiface':
    image_size = 192
    num_channels = 3
    total_train_size = 62327
else:
    raise NotImplementedError

batch_size = 128
num_epochs = 201

#dropout seems to be making it impossible to learn
#maybe only works for large nets
dropout_rate = 0.1
in_dropout_rate = 0.2
use_dropout = False

z_size = 100

use_batch_normalization = False
learning_rate = 0.0002 if not use_batch_normalization else 0.00005

gen_conv_ops = ['fulcon_in','conv_1','conv_2','conv_3']

if dataset_type=='cifar-10':
    #number of feature maps for each convolution layer
    gen_depth_conv = {'conv_1':256,'conv_2':128,'conv_3':64}

    noise_projection_shape = (4,4,512)
    #weights (conv): [width,height,in_depth,out_depth]
    #kernel (pool): [_,width,height,_]
    gen_conv_1_hyparams = {'weights':[5,5,gen_depth_conv['conv_1'],noise_projection_shape[2]],'stride':[1,2,2,1],'padding':'SAME','out_size':[batch_size,8,8,gen_depth_conv['conv_1']]}
    gen_conv_2_hyparams = {'weights':[5,5,gen_depth_conv['conv_2'],gen_depth_conv['conv_1']],'stride':[1,2,2,1],'padding':'SAME','out_size':[batch_size,16,16,gen_depth_conv['conv_2']]}
    gen_conv_3_hyparams = {'weights':[5,5,num_channels,gen_depth_conv['conv_2']],'stride':[1,2,2,1],'padding':'SAME','out_size':[batch_size,32,32,num_channels]}

    pool_large_hyperparams = {'type':'avg','kernel':[1,5,5,1],'stride':[1,1,1,1],'padding':'SAME'}

    # fully connected layer hyperparameters
    out_hyparams = {'in':0,'out':100}
    fulcon_in_hyparams = {'in':z_size,'out':noise_projection_shape[0]*noise_projection_shape[1]*noise_projection_shape[2]}

elif dataset_type=='wikiface':
    #number of feature maps for each convolution layer
    gen_depth_conv = {'conv_1':256,'conv_2':128,'conv_3':64}

    noise_projection_shape = (4,4,512)
    #weights (conv): [width,height,in_depth,out_depth]
    #kernel (pool): [_,width,height,_]
    gen_conv_1_hyparams = {'weights':[5,5,gen_depth_conv['conv_1'],noise_projection_shape[2]],'stride':[1,2,2,1],'padding':'SAME','out_size':[batch_size,8,8,gen_depth_conv['conv_1']]}
    gen_conv_2_hyparams = {'weights':[5,5,gen_depth_conv['conv_2'],gen_depth_conv['conv_1']],'stride':[1,4,4,1],'padding':'SAME','out_size':[batch_size,32,32,gen_depth_conv['conv_2']]}
    gen_conv_3_hyparams = {'weights':[5,5,num_channels,gen_depth_conv['conv_2']],'stride':[1,6,6,1],'padding':'SAME','out_size':[batch_size,192,192,num_channels]}

    pool_large_hyperparams = {'type':'avg','kernel':[1,5,5,1],'stride':[1,1,1,1],'padding':'SAME'}

    # fully connected layer hyperparameters
    out_hyparams = {'in':0,'out':100}
    fulcon_in_hyparams = {'in':z_size,'out':noise_projection_shape[0]*noise_projection_shape[1]*noise_projection_shape[2]}


gen_hyparams = {
    'fulcon_in':fulcon_in_hyparams,'conv_1': gen_conv_1_hyparams, 'conv_2': gen_conv_2_hyparams, 'conv_3':gen_conv_3_hyparams,
    'fulcon_out':out_hyparams,'pool_large':pool_large_hyperparams
}

gen_weights,gen_biases = {},{}
gen_BNgammas,gen_BNbetas = {},{}

#=============================================================================================================================
disc_conv_ops = ['conv_1','conv_2','conv_3','fulcon_out']

if dataset_type=='cifar-10':
    #number of feature maps for each convolution layer
    disc_depth_conv = {'conv_1':64,'conv_2':128,'conv_3':256}

    #weights (conv): [width,height,in_depth,out_depth]
    #kernel (pool): [_,width,height,_]
    disc_conv_1_hyparams = {'weights':[5,5,num_channels,disc_depth_conv['conv_1']],'stride':[1,2,2,1],'padding':'SAME'}
    disc_conv_2_hyparams = {'weights':[5,5,disc_depth_conv['conv_1'],disc_depth_conv['conv_2']],'stride':[1,2,2,1],'padding':'SAME'}
    disc_conv_3_hyparams = {'weights':[3,3,disc_depth_conv['conv_2'],disc_depth_conv['conv_3']],'stride':[1,2,2,1],'padding':'SAME'}

    # fully connected layer hyperparameters
    out_hyparams = {'in':0,'out':1}
    disc_hyparams = {
        'conv_1': disc_conv_1_hyparams, 'conv_2': disc_conv_2_hyparams, 'conv_3':disc_conv_3_hyparams, 'fulcon_out':out_hyparams
    }
elif dataset_type=='wikiface':
    disc_depth_conv = {'conv_1':64,'conv_2':128,'conv_3':256}

    #weights (conv): [width,height,in_depth,out_depth]
    #kernel (pool): [_,width,height,_]
    disc_conv_1_hyparams = {'weights':[5,5,num_channels,disc_depth_conv['conv_1']],'stride':[1,6,6,1],'padding':'SAME'} #192/6 = 32
    disc_conv_2_hyparams = {'weights':[5,5,disc_depth_conv['conv_1'],disc_depth_conv['conv_2']],'stride':[1,4,4,1],'padding':'SAME'} #32/4=8
    disc_conv_3_hyparams = {'weights':[3,3,disc_depth_conv['conv_2'],disc_depth_conv['conv_3']],'stride':[1,2,2,1],'padding':'SAME'} #8/2=4

    # fully connected layer hyperparameters
    out_hyparams = {'in':0,'out':1}
    disc_hyparams = {
        'conv_1': disc_conv_1_hyparams, 'conv_2': disc_conv_2_hyparams, 'conv_3':disc_conv_3_hyparams, 'fulcon_out':out_hyparams
    }

disc_weights,disc_biases = {},{}
disc_BNgammas,disc_BNbetas = {},{}

GenEMAMean,GenEMAVariance = {},{}
DiscEMAMean,DiscEMAVariance = {},{}

'''
===========================================================================
                Network Architecture (Generator)
---------------------------------------------------------------------------
  BN(outFC,sizeFC)        BN(outC1,depthC1)  BN(outC1,depthC1)
      V                              V             V
Z - out(FC) - Reshape(out(FC)) - out(conv_1) - out(conv_2) - out(conv_3,Tanh)
  ^                            ^             ^             ^
W(fulcon_in)                W(conv_1)     W(conv_2)    W(conv_3)

===========================================================================
                Network Architecture (Discriminator)
---------------------------------------------------------------------------
    BN(outGen,deptC1) BN(outC2,depthC2) BN(outC3,depthC3)
               V             V             V
out(GEN) - out(conv_1) - out(conv_2) - out(conv_3) - Reshape(out(conv_3) - out(FC,sigmoid)
         ^             ^             ^                                ^
     W(conv_1)     W(conv_2)     W(conv_3)                       W(fulcon_out)
===========================================================================
'''


'''=========================================================================
GENERATOR FUNCTIONS
========================================================================='''

def create_generator_layers():
    global GenEMAMean,GenEMAVariance
    print('Defining parameters ...')

    for op in gen_conv_ops:
        if 'fulcon_in' in op:
            print('\tDefining weights and biases for %s'%op)
            gen_weights[op]=tf.Variable(
                tf.truncated_normal([gen_hyparams[op]['in'],gen_hyparams[op]['out']],
                                    stddev=0.02),name='Genweights_'+op)
            gen_biases[op] = tf.Variable(tf.constant(0.0,shape=[gen_hyparams[op]['out']]), name='Genbias_'+op)

            if use_batch_normalization:
                # Gamma and Beta single value for a single feature map and a single batch
                gen_BNgammas[op] = tf.Variable(tf.truncated_normal([1,gen_hyparams[op]['out']],stddev=0.02),name='GenBatchNormGamma'+op)
                gen_BNbetas[op] = tf.Variable(tf.truncated_normal([1,gen_hyparams[op]['out']],stddev=0.02),name='GenBatchNormBeta'+op)

        if 'conv' in op:
            print('\tDefining weights and biases for %s (weights:%s)'%(op,gen_hyparams[op]['weights']))
            print('\t\tWeights:%s'%gen_hyparams[op]['weights'])
            print('\t\tBias:%d'%gen_hyparams[op]['weights'][3])
            gen_weights[op]=tf.Variable(
                tf.truncated_normal(gen_hyparams[op]['weights'],
                                    stddev=0.02),name='Genweights_'+op)
            gen_biases[op] = tf.Variable(tf.constant(0.0,shape=[gen_hyparams[op]['weights'][2]]), name='Genbias_'+op)

            if use_batch_normalization:
                # Gamma and Beta single value for a single feature map and a single batch
                gen_BNgammas[op] = tf.Variable(tf.truncated_normal([1,1,1,gen_hyparams[op]['weights'][2]],stddev=0.02),name='GenBatchNormGamma'+op)
                gen_BNbetas[op] = tf.Variable(tf.truncated_normal([1,1,1,gen_hyparams[op]['weights'][2]],stddev=0.02),name='GenBatchNormBeta'+op)

def batch_norm_gen(x, batch_size, op, is_train,decay):
    global GenEMAMean,GenEMAVariance

    if 'conv' in op:
        # Fon conv layers, we want beta and gamma per FEATURE MAP not Activation
        # X shape (batch_size,x,y,depth)
        x_mean,x_var = tf.nn.moments(x,[0,1,2],keep_dims=True)
    else:
        # For fully connected, we want beta and gamma per activation
        # X shape (batch_size,activation_size)
        x_mean,x_var = tf.nn.moments(x,[0],keep_dims=True)

    if is_train:
        with tf.control_dependencies([GenEMAMean[op].assign((1.0-decay)*GenEMAMean[op]+decay*x_mean),
                                      GenEMAVariance[op].assign((1.0-decay)*GenEMAVariance[op]+decay*x_var)]):
            x_hat = (x - x_mean)/(tf.sqrt(x_var+epsilon))
            y = gen_BNgammas[op]*x_hat + gen_BNbetas[op]
            return y
    else:
        glb_mean,glb_var = GenEMAMean[op],batch_size*GenEMAVariance[op]/(batch_size-1)
        y = (gen_BNgammas[op]/tf.sqrt(glb_var+epsilon))*x +(gen_BNbetas[op]-(gen_BNgammas[op]*glb_mean/tf.sqrt(glb_var+epsilon)))
        return y

def get_generator_output(dataset,is_train):

    # Variables.

    x = dataset
    #if use_batch_normalization:
        #x = batch_norm_gen(x,batch_size,'input',is_train)

    print('Calculating inputs for data X(%s)...'%x.get_shape().as_list())

    x = tf.matmul(x,gen_weights['fulcon_in'])
    if use_batch_normalization:
        if 'fulcon_in' not in GenEMAMean and 'fulcon_in' not in GenEMAVariance:
            x_shp = x.get_shape().as_list()
            GenEMAMean['fulcon_in'] = tf.Variable(tf.zeros([1,x_shp[1]],dtype=tf.float32),name='GEMAmean_fulcon_in')
            GenEMAVariance['fulcon_in'] = tf.Variable(tf.ones([1,x_shp[1]],dtype=tf.float32),name='GEMAvariance_fulcon_in')

        x = batch_norm_gen(x,batch_size,'fulcon_in',is_train,bn_decay)

    x = tf.nn.relu(x+gen_biases['fulcon_in'])
    x = tf.reshape(x,[batch_size,noise_projection_shape[0],noise_projection_shape[1],noise_projection_shape[2]])
    print('Size of X after reshaping (%s)...'%x.get_shape().as_list())

    for op in gen_conv_ops:
        if 'conv' in op:
            print('\tTranspose Covolving data (%s)'%op)
            x = tf.nn.conv2d_transpose(x, gen_weights[op], gen_hyparams[op]['out_size'], gen_hyparams[op]['stride'], padding=gen_hyparams[op]['padding'])

            if op != gen_conv_ops[-1]:
                # The paper says it is enough to perform normalization on Wx instead of Wx+b
                # because b cancels out anyway
                if use_batch_normalization:
                    if op not in GenEMAMean and op not in GenEMAVariance:
                        x_shp = x.get_shape().as_list()
                        GenEMAMean[op] = tf.Variable(tf.zeros([1,1,1,x_shp[3]],dtype=tf.float32),name='GEMAmean_'+op)
                        GenEMAVariance[op] = tf.Variable(tf.ones([1,1,1,x_shp[3]],dtype=tf.float32),name='GEMAvariance_'+op)

                    # X shape (batch_size,x,y,depth)
                    x = batch_norm_gen(x,batch_size,op,is_train,bn_decay)
                x = tf.nn.relu(x + gen_biases[op])
            else:
                print('\tUsing Tanh for last Layer %s'%op)
                x = tf.nn.tanh(x + gen_biases[op])
            print('\t\tX after %s:%s'%(op,x.get_shape().as_list()))
        if 'pool' in op:
            print('\tPooling with large kernel (%s)'%op)
            x = tf.nn.avg_pool(x,ksize=gen_hyparams[op]['kernel'],strides=gen_hyparams[op]['stride'],padding=gen_hyparams[op]['padding'])
            print('\t\tX after %s:%s'%(op,x.get_shape().as_list()))

    return x

def calc_generator_loss(DoutFake):
    # Training computation.
    #loss = -tf.reduce_mean(tf.log(disc_out))
    Gloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(DoutFake, tf.ones_like(DoutFake)))
    return Gloss

def optimize_generator(Gloss,Gvariables):
    # Optimizer.
    Goptimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.5)
    grads = Goptimizer.compute_gradients(Gloss, var_list=Gvariables)
    minimize_Gopt = Goptimizer.minimize(Gloss, var_list=Gvariables)
    #grad = optimizer.compute_gradients(gen_loss,variables)
    #update = optimizer.apply_gradients(grad)
    return minimize_Gopt,grads

'''==================================================================
DISCRIMINATOR FUNCTIONS
=================================================================='''
def create_discriminator_layers():
    global DiscEMAMean,DiscEMAVariance
    print('Defining parameters ...')

    for op in disc_conv_ops:
        if 'fulcon' in op:
            #we don't create weights biases for fully connected layers because we need to calc the
            #fan_out of the last convolution/pooling (subsampling) layer
            #as that's gonna be fan_in for the 1st hidden layer
            break
        if 'conv' in op:
            print('\tDefining weights and biases for %s (weights:%s)'%(op,disc_hyparams[op]['weights']))
            print('\t\tWeights:%s'%disc_hyparams[op]['weights'])
            print('\t\tBias:%d'%disc_hyparams[op]['weights'][3])
            disc_weights[op]=tf.Variable(
                tf.truncated_normal(disc_hyparams[op]['weights'],
                                    stddev=0.02
                                    ),name='Discweights_'+op
            )
            disc_biases[op] = tf.Variable(tf.constant(0.0,shape=[disc_hyparams[op]['weights'][3]]),name='Discbias_'+op)

            if use_batch_normalization:
                disc_BNgammas[op] = tf.Variable(tf.truncated_normal([1,1,1,disc_hyparams[op]['weights'][3]],stddev=0.02),name='DiscBatchNormGamma'+op)
                disc_BNbetas[op] = tf.Variable(tf.truncated_normal([1,1,1,disc_hyparams[op]['weights'][3]],stddev=0.02),name='DiscBatchNormBeta'+op)


def update_discriminator_fulcon_in(fan_in):
    for op in disc_conv_ops:
        if 'fulcon_out' in op:
            disc_hyparams[op]['in'] = fan_in
            break

def add_discriminator_fulcon_out():
    if 'fulcon_out' not in disc_weights:
        disc_weights['fulcon_out'] = tf.Variable(
                    tf.truncated_normal([disc_hyparams['fulcon_out']['in'],disc_hyparams['fulcon_out']['out']],
                                        stddev=0.02),name='DiscWeights_fulcon_out')
        disc_biases['fulcon_out'] = tf.Variable(tf.constant(0.0,shape=[disc_hyparams['fulcon_out']['out']]),name='DiscWeights_fulcon_out')

def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)

def batch_norm_disc(x, batch_size, op, is_train, decay):
    global DiscEMAMean,DiscEMAVariance
    # X shape (batch_size,x,y,depth)
    if 'conv' in op:
        x_mean,x_var = tf.nn.moments(x,[0,1,2],keep_dims=True)
    else:
        x_mean,x_var = tf.nn.moments(x,[0],keep_dims=True)

    if is_train:
        with tf.control_dependencies([DiscEMAMean[op].assign((1.0-decay)*DiscEMAMean[op]+decay*x_mean),
                                      DiscEMAVariance[op].assign((1.0-decay)*DiscEMAVariance[op]+decay*x_var)]):
            x_hat = (x - x_mean)/(tf.sqrt(x_var)+epsilon)
            y = disc_BNgammas[op]*x_hat + disc_BNbetas[op]
            return y
    else:
        glb_mean,glb_var = DiscEMAMean[op],batch_size*DiscEMAVariance[op]/(batch_size-1)
        y = (disc_BNgammas[op]/tf.sqrt(glb_var+epsilon))*x +(disc_BNbetas[op]-(disc_BNgammas[op]*glb_mean/tf.sqrt(glb_var+epsilon)))
        return y

def get_discriminator_output(dataset,is_train):

    # Variables.
    x = dataset
    #if use_batch_normalization:
        # X shape (batch_size,x,y,depth)
        #x = batch_norm_disc(x, batch_size, 'input', is_train)

    print('Calculating inputs for data X(%s)...'%x.get_shape().as_list())
    for op in disc_conv_ops:
        if 'conv' in op:
            print('\tCovolving data (%s)'%op)
            x = tf.nn.conv2d(x, disc_weights[op], disc_hyparams[op]['stride'], padding=disc_hyparams[op]['padding'])
            # The paper says it is enough to perform normalization on Wx instead of Wx+b
            # because b cancels out anyway
            if use_batch_normalization:
                if op not in DiscEMAMean and op not in DiscEMAVariance:
                    x_shp = x.get_shape().as_list()
                    DiscEMAMean[op] = tf.Variable(tf.zeros([1,1,1,x_shp[3]],dtype=tf.float32),name='DEMAmean_'+op)
                    DiscEMAVariance[op] = tf.Variable(tf.ones([1,1,1,x_shp[3]],dtype=tf.float32),name='DEMAvariance_'+op)
                # X shape (batch_size,x,y,depth)
                x = batch_norm_disc(x, batch_size, op, is_train,bn_decay)

            x = lrelu(x + disc_biases[op])
            print('\t\tX after %s:%s'%(op,x.get_shape().as_list()))

        if 'fulcon' in op:
            break

    # we need to reshape the output of last subsampling layer to
    # convert 4D output to a 2D input to the hidden layer
    # e.g subsample layer output [batch_size,width,height,depth] -> [batch_size,width*height*depth]
    shape = x.get_shape().as_list()
    rows = shape[0]
    update_discriminator_fulcon_in(shape[1] * shape[2] * shape[3])
    add_discriminator_fulcon_out()

    for op in disc_conv_ops:
        if 'fulcon' in op:
            print('Unwrapping last convolution layer %s to %s hidden layer'%(shape,(rows,disc_hyparams[op]['in'])))
            x = tf.reshape(x, [rows,disc_hyparams[op]['in']])
            x = tf.matmul(x,disc_weights['fulcon_out'])
            #if use_batch_normalization:
                #x = batch_norm_disc(x, batch_size, op, is_train)

            break

    return tf.nn.sigmoid(x + disc_biases['fulcon_out']),x + disc_biases['fulcon_out']

def calc_discriminator_loss(DLogitReal,DLogitFake):
    # Training computation.
    # e.g. log(0.2) = -0.7, log(1) = 0
    #loss = -tf.reduce_mean(tf.log(disc_out_with_real + loss_epsilon) + tf.log(1.-disc_out_with_gen + loss_epsilon))
    Dloss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(DLogitReal, tf.ones_like(DLogitReal)))\
           + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(DLogitFake, tf.zeros_like(DLogitFake)))

    return Dloss

def optimize_discriminator(Dloss,Dvariables):
    # Optimizer.
    #Doptimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(Dloss, var_list=Dvariables)
    Doptimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.5)
    minimize_Dopt = Doptimizer.minimize(Dloss, var_list=Dvariables)
    grad = Doptimizer.compute_gradients(Dloss,Dvariables)

    return minimize_Dopt,grad

def save_images(image, image_path):
    scipy.misc.imsave(image_path, inverse_transform(image))

def inverse_transform(images):
    return (images+1.)/2.


# OBSERVATIONS
# DISC loss goes NaN at the beginning
# Later DISC loss goes negative later
if __name__=='__main__':

    global train_size,valid_size,test_size
    global log_suffix,data_filename
    global total_iterations

    debug_grads = False
    try:
        opts,args = getopt.getopt(
            sys.argv[1:],"",['log_suffix='])
    except getopt.GetoptError as err:
        print('')

    if len(opts)!=0:
        for opt,arg in opts:
            if opt == '--log_suffix':
                log_suffix = arg

    image_save_directory='images_dir'
    if dataset_type=='cifar-10':
        #load_data.load_and_save_data_cifar10('cifar-10.pickle',zca_whiten=False)
        (full_train_dataset,full_train_labels),(valid_dataset,valid_labels),(test_dataset,test_labels)=load_data.reformat_data_cifar10('cifar-10.pickle')
    if dataset_type=='wikiface':
        #load_data.load_and_save_data_cifar10('cifar-10.pickle',zca_whiten=False)
        fp1 = np.memmap('/home/tgan4199/DCGAN'+os.sep+'data'+os.sep+'wiki_faces'+os.sep+'wikiface_dataset', dtype=np.float32,mode='r',
                                           offset=np.dtype('float32').itemsize*1,shape=(batch_size*200,image_size,image_size,num_channels))
        full_train_dataset = fp1[:,:,:,:]

    graph = tf.Graph()

    # Logging generator and discriminator loss
    loss_logger = logging.getLogger('loss_logger_'+log_suffix)
    loss_logger.setLevel(logging.INFO)
    lossFileHandler = logging.FileHandler('loss_logger_'+log_suffix, mode='w')
    lossFileHandler.setFormatter(logging.Formatter('%(message)s'))
    loss_logger.addHandler(lossFileHandler)
    loss_logger.info('#Epoch, Iteration, Gen Loss, Disc Loss')

    if debug_grads:
        # Gradient logger
        grad_logger = logging.getLogger('grad_logger_'+log_suffix)
        grad_logger.setLevel(logging.INFO)
        gradFileHandler = logging.FileHandler('grad_logger_'+log_suffix, mode='w')
        gradFileHandler.setFormatter(logging.Formatter('%(message)s'))
        grad_logger.addHandler(gradFileHandler)

        # Gradient Stat logger
        gradStat_logger = logging.getLogger('gradStat_logger_'+log_suffix)
        gradStat_logger.setLevel(logging.INFO)
        gradStatFileHandler = logging.FileHandler('gradStat_logger_'+log_suffix, mode='w')
        gradStatFileHandler.setFormatter(logging.Formatter('%(message)s'))
        gradStat_logger.addHandler(gradStatFileHandler)
        gradStat_logger.info('#Epoch,Iteration,Min,Max,flow_percentage')

    test_accuracies = []
    total_iterations = 0

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    with tf.Session(graph=graph,
                    config=tf.ConfigProto(allow_soft_placement=True,gpu_options=gpu_options)) as session, \
            tf.device('/gpu:0'):
        #tf.global_variables_initializer().run()
        # Input data.

        print('Input data defined...\n')
        tf_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))

        tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size,image_size,image_size,num_channels))


        train_dataset = full_train_dataset


        train_size = train_dataset.shape[0]

        # Input data.
        print('Running with %d data points...\n'%train_size)
        tf_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels),name='train_dataset')

        tf_noise_dataset = tf.placeholder(tf.float32, shape=(batch_size,z_size),name='noise_dataset')

        tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size,image_size,image_size,num_channels),name='test_dataset')

        global_step = tf.Variable(0, trainable=False)
        #start_lr = tf.Variable(start_lr)

        create_generator_layers()
        create_discriminator_layers()

        gen_dataset = get_generator_output(tf_noise_dataset,True)
        gen_images_fn = get_generator_output(tf_noise_dataset,False)

        DoutFakeTF,DLogitFakeTF = get_discriminator_output(gen_dataset,True)
        DoutRealTF,DLogitRealTF  = get_discriminator_output(tf_dataset,True)

        GLoss = calc_generator_loss(DLogitFakeTF)
        DLoss = calc_discriminator_loss(DLogitRealTF,DLogitFakeTF)
        DLossSwapped = calc_discriminator_loss(DLogitFakeTF,DLogitRealTF)

        tvars = tf.trainable_variables()

        gVars = [v for v in tvars if v.name.startswith('Gen')]
        dVars = [v for v in tvars if v.name.startswith('Disc')]
        print('='*80)
        print('Variables')
        print('Generator Trainable Variables')
        print([v.name for v in gVars])
        print()
        print('Discriminator Trainable Variables')
        print([v.name for v in dVars])
        print('='*80)
        print()

        # Grad is a list of (gradient,variable) tuples for each variable in the passed variable list
        GenOptimizeTF,GgradsTF = optimize_generator(GLoss,gVars)
        DiscOptimizerTF,DgradsTF = optimize_discriminator(DLoss,dVars)
        DiscOptimizerSwappedTF,Dswap_gradsTF = optimize_discriminator(DLossSwapped,dVars)

        tf.initialize_all_variables().run()

        print('Initialized...')
        print('\tBatch size:',batch_size)
        print('\tNum Epochs: ',num_epochs)
        print('\tDropout: ',use_dropout,', ',dropout_rate)
        print('='*80)
        print()

        accuracy_drop = 0 # used for early stopping
        max_test_accuracy = 0
        gen_images = None
        prevDoutFakeNoSig = None
        prevDoutRealNoSig = None

        lg,ld = None,None
        DoutFake,DoutFakeNoSigmoid,DoutReal,DoutRealNoSigmoid = 0,0,0,0

        gen_losses,disc_losses = [],[]
        do_Doptimize = True
        GoptCount,DoptCount = 0,0
        for epoch in range(num_epochs):
            for iteration in range(floor(float(train_size)/batch_size)):
                offset = iteration * batch_size
                assert offset < train_size
                batch_data = train_dataset[offset:offset + batch_size, :, :, :]

                z_data = np.random.uniform(-1.0,1.0,size=[batch_size,z_size]).astype(np.float32)

                gen_feed_dict = {tf_noise_dataset : z_data}
                _, DoutFake, DoutFakeNoSigmoid, lg = session.run([gen_dataset,DoutFakeTF,DLogitFakeTF,GLoss],
                                                                 feed_dict=gen_feed_dict)
                gen_losses.append(lg)

                disc_feed_dict = {tf_dataset : batch_data, tf_noise_dataset : z_data}
                DoutReal,DoutRealNoSigmoid, ld = session.run([DoutRealTF,DLogitRealTF,DLoss],
                                                             feed_dict=disc_feed_dict)
                disc_losses.append(ld)

                for _ in range(2):
                    session.run([GenOptimizeTF], feed_dict=gen_feed_dict)
                    GoptCount += 1
                #assert not np.isnan(lg)

                if do_Doptimize:
                    if np.random.random()<0.5/(epoch+1):
                        session.run([DiscOptimizerSwappedTF], feed_dict=disc_feed_dict)
                        DoptCount += 1
                    else:
                        session.run([DiscOptimizerTF], feed_dict=disc_feed_dict)
                        DoptCount += 1
                #assert not np.isnan(ld)

                if ld<0 or lg<0:
                    length = 20
                    print('Detected Negative Losses %.3f, %.3f'%(ld,lg))
                    print('Dout for Fake data')
                    print(DoutFake[:length])
                    print()
                    print(DoutFakeNoSigmoid[:length])
                    print('Dout for Real data')
                    print(DoutReal[:length])
                    print()
                    print(DoutRealNoSigmoid[:length])

                if np.any(np.isnan(DoutFake)) or np.any(np.isnan(DoutReal)):
                    length = 20
                    print('Detected NaN')
                    if prevDoutFakeNoSig is not None:
                        print('Previous Discriminator (Fake) out (',epoch,' ',iteration,': ',prevDoutFakeNoSig[:length])
                        print()
                    print('Discriminator (Fake) out (',epoch,' ',iteration,': ',DoutFakeNoSigmoid[:length])
                    print()
                    print('Discriminator (Fake) Sigmoid out (',epoch,' ',iteration,': ',DoutFake[:length])
                    print()
                    if prevDoutRealNoSig is not None:
                        print('Previous Discriminator (Real) out (',epoch,' ',iteration,': ',prevDoutRealNoSig[:length])
                        print()
                    print('Discriminator (Real) out (',epoch,' ',iteration,': ',DoutRealNoSigmoid[:length])
                    print()
                    print('Discriminator (Real) Sigmoid out (',epoch,' ',iteration,': ',DoutReal[:length])

                #assert not np.any(ld<0) and not np.any(lg<0)
                #assert not np.any(np.isnan(DoutFake))
                #assert not np.any(np.isnan(DoutReal))
                if total_iterations % 1 == 0:

                    loss_logger.info('%d,%d,%.5f,%.5f',epoch,iteration,np.mean(gen_losses),np.mean(disc_losses))
                    print('Minibatch GEN loss (%.3f) and DISC loss (%.3f) epoch,iteration %d,%d' % (np.mean(gen_losses),np.mean(disc_losses),epoch,iteration))
                    print('\tGopt, Dopt: %d, %d'%(GoptCount,DoptCount))
                    GoptCount,DoptCount = 0,0
                    gen_losses,disc_losses = [],[]

                    if debug_grads:
                        Ggrads = session.run([g for g,v in GgradsTF if g is not None], feed_dict=gen_feed_dict)
                        Ggrads = zip([v.name for g,v in GgradsTF if g is not None],Ggrads)
                        Dgrads = session.run([g for g,v in DgradsTF if g is not None], feed_dict=disc_feed_dict)
                        Dgrads = zip([v.name for g,v in DgradsTF if g is not None],Dgrads)

                        grad_logger.info('#Generator Gradient')
                        for v,g in Ggrads:
                            gradStat_logger.info('Gen,%s,%d,%d,%.5f,%.5f,%.2f',v,epoch,iteration,np.min(g.flatten()),np.max(g.flatten()),
                                                 len(list(np.where(g.flatten()>0.001)[0]))*100.0/g.flatten().size)
                            grad_string = ''
                            for val in g.flatten()[:100]:
                                grad_string += '%.3f,'%val
                            grad_logger.info('%s,%s',v,grad_string)

                        grad_logger.info('#Discriminator Gradient')
                        for v,g in Dgrads:
                            gradStat_logger.info('Disc,%s,%d,%d,%.5f,%.5f,%.2f',v,epoch,iteration,np.min(g.flatten()),np.max(g.flatten()),
                                                 len(list(np.where(g.flatten()>0.001)[0]))*100.0/g.flatten().size)
                            grad_string = ''
                            for val in g.flatten()[:100]:
                                grad_string += '%.3f,'%val
                            grad_logger.info('%s,%s',v,grad_string)

                total_iterations += 1
                prevDoutFakeNoSig = DoutFakeNoSigmoid
                prevDoutRealNoSig = DoutRealNoSigmoid

            if epoch%5==0:

                z_data = np.random.uniform(-1.0,1.0,size=[batch_size,z_size]).astype(np.float32)
                gen_feed_dict = {tf_noise_dataset : z_data}
                gen_images = session.run([gen_images_fn], feed_dict=gen_feed_dict)
                for index,img in enumerate(batch_data[:10]):
                    if index<2:
                        print('Image Shape: %s'%str(img.shape))
                    filename = image_save_directory+os.sep+'real_'+str(epoch)+'_'+str(index)+'.png'
                    save_images(img,filename)

                for index in range(10):
                    if index<2:
                        print('Image Shape: %s'%str(gen_images[0][index].shape))
                    filename = image_save_directory+os.sep+'gen_'+str(epoch)+'_'+str(index)+'.png'
                    save_images(gen_images[0][index],filename)

