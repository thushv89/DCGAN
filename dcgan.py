__author__ = 'Thushan Ganegedara'

import tensorflow as tf
import numpy as np
from math import ceil,floor
import load_data
import scipy.misc
import scipy
import os


epsilon = 1e-8
loss_epsilon = 1e-8

dataset_type = 'cifar-10'
# hyperparameters
if dataset_type=='cifar-10':
    image_size = 32
    num_labels = 10
    num_channels = 3 # rgb
elif dataset_type=='imagenet':
    image_size = 224
    num_labels = 100
    num_channels = 3 # grayscale
else:
    raise NotImplementedError

batch_size = 128
num_epochs = 250
start_lr = 0.2

#dropout seems to be making it impossible to learn
#maybe only works for large nets
dropout_rate = 0.1
in_dropout_rate = 0.2
use_dropout = False

z_size = 100

use_batch_normalization = True
learning_rate = 0.0002 if not use_batch_normalization else 0.003

gen_conv_ops = ['fulcon_in','conv_1','conv_2','conv_3']

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
hidden_1_hyparams = {'in':0,'out':1024}
hidden_2_hyparams = {'in':1024,'out':512}
out_hyparams = {'in':0,'out':100}
fulcon_in_hyparams = {'in':z_size,'out':noise_projection_shape[0]*noise_projection_shape[1]*noise_projection_shape[2]}

gen_hyparams = {
    'fulcon_in':fulcon_in_hyparams,'conv_1': gen_conv_1_hyparams, 'conv_2': gen_conv_2_hyparams, 'conv_3':gen_conv_3_hyparams,
    'fulcon_hidden_1':hidden_1_hyparams,'fulcon_hidden_2': hidden_2_hyparams, 'fulcon_out':out_hyparams,'pool_large':pool_large_hyperparams
}

gen_weights,gen_biases = {},{}
gen_BNgammas,gen_BNbetas = {},{}

#=============================================================================================================================
disc_conv_ops = ['conv_1','conv_2','conv_3','fulcon_out']

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

disc_weights,disc_biases = {},{}
disc_BNgammas,disc_BNbetas = {},{}

def create_generator_layers():
    print('Defining parameters ...')

    if use_batch_normalization:
        gen_BNgammas['input'] = tf.Variable(tf.truncated_normal([1,z_size],stddev=0.02),name='GenBatchNormGamma_input')
        gen_BNbetas['input'] = tf.Variable(tf.truncated_normal([1,z_size],stddev=0.02),name='GenBatchNormBeta_input')

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


def batch_norm_gen(x, batch_size, op, is_train):
    # X shape (batch_size,x,y,depth)
    if 'conv' in op:
        x_mean,x_var = tf.nn.moments(x,[1,2,3],keep_dims=True)
    else:
        x_mean,x_var = tf.nn.moments(x,[1],keep_dims=True)

    if is_train:
        x_hat = (x - x_mean)/(tf.sqrt(x_var)+epsilon)
        y = gen_BNgammas[op]*x_hat + gen_BNbetas[op]
        return y
    else:
        x_var = batch_size*x_var/(batch_size-1)
        y = (gen_BNgammas[op]/tf.sqrt(x_var+epsilon))*x +(gen_BNbetas[op]-(gen_BNgammas[op]*x_mean/tf.sqrt(x_var+epsilon)))
        return y

def get_generator_output(dataset,is_train):

    # Variables.

    x = dataset
    #if use_batch_normalization:
        #x = batch_norm_gen(x,batch_size,'input',is_train)

    print('Calculating inputs for data X(%s)...'%x.get_shape().as_list())

    x = tf.matmul(x,gen_weights['fulcon_in'])
    if use_batch_normalization:
        x = batch_norm_gen(x,batch_size,'fulcon_in',is_train)

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
                    # X shape (batch_size,x,y,depth)
                    x = batch_norm_gen(x,batch_size,op,is_train)
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
    Goptimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.5).minimize(Gloss, var_list=Gvariables)
    #grad = optimizer.compute_gradients(gen_loss,variables)
    #update = optimizer.apply_gradients(grad)
    return Goptimizer

'''==================================================================
DISCRIMINATOR FUNCTIONS
=================================================================='''
def create_discriminator_layers():
    print('Defining parameters ...')

    if use_batch_normalization:
        disc_BNgammas['input'] = tf.Variable(tf.truncated_normal([1,image_size,image_size,num_channels],stddev=0.02),name='DiscBatchNormGamma_input')
        disc_BNbetas['input'] = tf.Variable(tf.truncated_normal([1,image_size,image_size,num_channels],stddev=0.02),name='DiscBatchNormBeta_input')

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

def batch_norm_disc(x, batch_size, op, is_train):
    # X shape (batch_size,x,y,depth)
    if 'conv' in op:
        x_mean,x_var = tf.nn.moments(x,[1,2,3],keep_dims=True)
    else:
        x_mean,x_var = tf.nn.moments(x,[1],keep_dims=True)

    if is_train:
        x_hat = (x - x_mean)/(tf.sqrt(x_var)+epsilon)
        y = disc_BNgammas[op]*x_hat + disc_BNbetas[op]
        return y
    else:
        x_var = batch_size*x_var/(batch_size-1)
        y = (disc_BNgammas[op]/tf.sqrt(x_var+epsilon))*x +(disc_BNbetas[op]-(disc_BNgammas[op]*x_mean/tf.sqrt(x_var+epsilon)))
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
                # X shape (batch_size,x,y,depth)
                x = batch_norm_disc(x, batch_size, op, is_train)

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
    Doptimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,beta1=0.5).minimize(Dloss, var_list=Dvariables)
    #grad = optimizer.compute_gradients(loss,variables)
    #update = optimizer.apply_gradients(grad)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return Doptimizer

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

    image_save_directory='images_dir'
    if dataset_type=='cifar-10':
        #load_data.load_and_save_data_cifar10('cifar-10.pickle',zca_whiten=False)
        (full_train_dataset,full_train_labels),(valid_dataset,valid_labels),(test_dataset,test_labels)=load_data.reformat_data_cifar10('cifar-10.pickle')

    graph = tf.Graph()

    # Value logger will log info used to calculate policies
    '''test_logger = logging.getLogger('test_logger_'+log_suffix)
    test_logger.setLevel(logging.INFO)
    fileHandler = logging.FileHandler('test_logger_'+log_suffix, mode='w')
    fileHandler.setFormatter(logging.Formatter('%(message)s'))
    test_logger.addHandler(fileHandler)'''

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
        tf_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

        tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size,image_size,image_size,num_channels))
        tf_test_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

        train_dataset = full_train_dataset
        train_labels = full_train_labels

        train_size,valid_size,test_size = train_dataset.shape[0],valid_dataset.shape[0],test_dataset.shape[0]

        # Input data.
        print('Running with %d data points...\n'%train_size)
        tf_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels),name='train_dataset')
        tf_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels),name='train_labels')

        tf_noise_dataset = tf.placeholder(tf.float32, shape=(batch_size,z_size),name='noise_dataset')

        tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size,image_size,image_size,num_channels),name='test_dataset')
        tf_test_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels),name='test_labels')

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
        print([v.name for v in tvars])

        gVars = [v for v in tvars if v.name.startswith('Gen')]
        dVars = [v for v in tvars if v.name.startswith('Disc')]
        assert len(gVars)+len(dVars)==len(tvars)

        GenOptimizeTF = optimize_generator(GLoss,gVars)
        DiscOptimizerTF = optimize_discriminator(DLoss,dVars)
        DiscOptimizerSwappedTF = optimize_discriminator(DLossSwapped,dVars)

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
                batch_labels = train_labels[offset:offset + batch_size, :]

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
                assert not np.isnan(lg)

                if do_Doptimize:
                    if np.random.random()<0.1:
                        session.run([DiscOptimizerSwappedTF], feed_dict=disc_feed_dict)
                        DoptCount += 1
                    else:
                        if np.random.random()<0.9:
                            session.run([DiscOptimizerTF], feed_dict=disc_feed_dict)
                            DoptCount += 1
                assert not np.isnan(ld)

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

                '''if (DoutFake and np.any(np.isnan(DoutFake))) or (DoutReal and np.any(np.isnan(DoutReal))):
                    length = 20
                    print('Detected NaN')
                    print('Previous Discriminator (Fake) out (',epoch,' ',iteration,': ',prevDoutFakeNoSig[:length])
                    print()
                    print('Discriminator (Fake) out (',epoch,' ',iteration,': ',DoutFakeNoSigmoid[:length])
                    print()
                    print('Discriminator (Fake) Sigmoid out (',epoch,' ',iteration,': ',DoutFake[:length])
                    print()
                    print('Previous Discriminator (Real) out (',epoch,' ',iteration,': ',prevDoutRealNoSig[:length])
                    print()
                    print('Discriminator (Real) out (',epoch,' ',iteration,': ',DoutRealNoSigmoid[:length])
                    print()
                    print('Discriminator (Real) Sigmoid out (',epoch,' ',iteration,': ',DoutReal[:length])'''

                assert not np.any(ld<0) and not np.any(lg<0)
                assert not np.any(np.isnan(DoutFake))
                assert not np.any(np.isnan(DoutReal))
                if total_iterations % 100 == 0:
                    if np.mean(disc_losses)<0.15 and np.mean(gen_losses)>1.0:
                        do_Doptimize= False
                    else:
                        do_Doptimize=True
                    print('Minibatch GEN loss (%.3f) and DISC loss (%.3f) epoch,iteration %d,%d' % (np.mean(gen_losses),np.mean(disc_losses),epoch,iteration))
                    print('\tGopt, Dopt: %d, %d'%(GoptCount,DoptCount))
                    GoptCount,DoptCount = 0,0
                    gen_losses,disc_losses = [],[]

                total_iterations += 1
                prevDoutFakeNoSig = DoutFakeNoSigmoid
                prevDoutRealNoSig = DoutRealNoSigmoid

            if epoch%5==0:

                z_data = np.random.uniform(-1.0,1.0,size=[batch_size,z_size]).astype(np.float32)
                gen_feed_dict = {tf_noise_dataset : z_data}
                gen_images = session.run([gen_images_fn], feed_dict=gen_feed_dict)
                '''for index,img in enumerate(batch_data[:10]):
                    if index<2:
                        print('Image Shape: %s'%str(img.shape))
                    filename = image_save_directory+os.sep+'real_'+str(epoch)+'_'+str(index)+'.png'
                    save_images(img,filename)'''

                for index in range(10):
                    if index<2:
                        print('Image Shape: %s'%str(gen_images[0][index].shape))
                    filename = image_save_directory+os.sep+'gen_'+str(epoch)+'_'+str(index)+'.png'
                    save_images(gen_images[0][index],filename)

