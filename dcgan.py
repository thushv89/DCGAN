__author__ = 'Thushan Ganegedara'

import tensorflow as tf
import numpy as np
from math import ceil,floor

total_iterations = 0

dataset_type = 'imagenet'
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

batch_size = 16

num_epochs = 250
decay_step = 10

start_lr = 0.2

#dropout seems to be making it impossible to learn
#maybe only works for large nets
dropout_rate = 0.1
in_dropout_rate = 0.2
use_dropout = False

z_size = 100

gen_conv_ops = ['fulcon_in','conv_1','pool_1','loc_res_norm','conv_2','pool_2','loc_res_norm','conv_3','pool_1','loc_res_norm']

#number of feature maps for each convolution layer
gen_depth_conv = {'conv_1':128,'conv_2':96,'conv_3':64}

noise_projection_shape = (4,4,256)
#weights (conv): [width,height,in_depth,out_depth]
#kernel (pool): [_,width,height,_]
gen_conv_1_hyparams = {'weights':[5,5,gen_depth_conv['conv_1'],noise_projection_shape[2]],'stride':[1,2,2,1],'padding':'SAME'}
gen_conv_2_hyparams = {'weights':[5,5,gen_depth_conv['conv_2'],gen_depth_conv['conv_1']],'stride':[1,1,1,1],'padding':'SAME'}
gen_conv_3_hyparams = {'weights':[5,5,num_channels,gen_depth_conv['conv_2']],'stride':[1,1,1,1],'padding':'SAME'}

# fully connected layer hyperparameters
hidden_1_hyparams = {'in':0,'out':1024}
hidden_2_hyparams = {'in':1024,'out':512}
out_hyparams = {'in':124,'out':100}
fulcon_in_hyparams = {'in':z_size,'out':noise_projection_shape[0]*noise_projection_shape[1]*noise_projection_shape[2]}

gen_hyparams = {
    'fulcon_in':fulcon_in_hyparams,'conv_1': gen_conv_1_hyparams, 'conv_2': gen_conv_2_hyparams, 'conv_3':gen_conv_3_hyparams,
    'fulcon_hidden_1':hidden_1_hyparams,'fulcon_hidden_2': hidden_2_hyparams, 'fulcon_out':out_hyparams
}

gen_weights,gen_biases = {},{}

#=============================================================================================================================
disc_conv_ops = ['conv_1','pool_1','loc_res_norm','conv_2','pool_2','loc_res_norm','conv_3','pool_1','loc_res_norm','fulcon_out']

#number of feature maps for each convolution layer
disc_depth_conv = {'conv_1':64,'conv_2':96,'conv_3':128}

#weights (conv): [width,height,in_depth,out_depth]
#kernel (pool): [_,width,height,_]
disc_conv_1_hyparams = {'weights':[5,5,num_channels,disc_depth_conv['conv_1']],'stride':[1,2,2,1],'padding':'SAME'}
disc_conv_2_hyparams = {'weights':[5,5,disc_depth_conv['conv_1'],disc_depth_conv['conv_2']],'stride':[1,1,1,1],'padding':'SAME'}
disc_conv_3_hyparams = {'weights':[5,5,disc_depth_conv['conv_2'],disc_depth_conv['conv_3']],'stride':[1,1,1,1],'padding':'SAME'}

# fully connected layer hyperparameters
out_hyparams = {'in':0,'out':1}
disc_hyparams = {
    'fulcon_in':fulcon_in_hyparams,'conv_1': disc_conv_1_hyparams, 'conv_2': disc_conv_2_hyparams, 'conv_3':disc_conv_3_hyparams,
    'fulcon_hidden_1':hidden_1_hyparams,'fulcon_hidden_2': hidden_2_hyparams, 'fulcon_out':out_hyparams
}

disc_weights,disc_biases = {},{}

def create_generator_layers():
    print('Defining parameters ...')

    for op in gen_conv_ops:
        if 'fulcon_in' in op:
            print('\tDefining weights and biases for %s (weights:%s)'%(op,gen_hyparams[op]['weights']))
            print('\t\tWeights:%s'%gen_hyparams[op]['weights'])
            print('\t\tBias:%d'%gen_hyparams[op]['weights'][3])
            gen_weights[op]=tf.Variable(
                tf.truncated_normal([gen_hyparams[op]['in'],gen_hyparams[op]['out']],
                                    stddev=2./min(5,gen_hyparams[op]['weights'][0])
                                    )
            )
            gen_biases[op] = tf.Variable(tf.constant(np.random.random()*0.001,shape=[gen_hyparams[op]['out']]))
        if 'conv' in op:
            print('\tDefining weights and biases for %s (weights:%s)'%(op,gen_hyparams[op]['weights']))
            print('\t\tWeights:%s'%gen_hyparams[op]['weights'])
            print('\t\tBias:%d'%gen_hyparams[op]['weights'][3])
            gen_weights[op]=tf.Variable(
                tf.truncated_normal(gen_hyparams[op]['weights'],
                                    stddev=2./min(5,gen_hyparams[op]['weights'][0])
                                    )
            )
            gen_biases[op] = tf.Variable(tf.constant(np.random.random()*0.001,shape=[gen_hyparams[op]['weights'][3]]))


def get_generator_output(dataset,is_train):

    # Variables.
    if not use_dropout:
        x = dataset
    else:
        x = tf.nn.dropout(dataset,1.0 - in_dropout_rate,seed=tf.set_random_seed(98765))
    print('Calculating inputs for data X(%s)...'%x.get_shape().as_list())

    x = tf.nn.relu(tf.matmul(x,gen_weights['fulcon_in'])+gen_biases['fulcon_in'])
    x = tf.reshape(x,[batch_size,noise_projection_shape[0],noise_projection_shape[1],noise_projection_shape[2]])

    for op in gen_conv_ops:
        if 'conv' in op:
            print('\tCovolving data (%s)'%op)
            x = tf.nn.conv2d_transpose(x, gen_weights[op], gen_hyparams[op]['stride'], padding=gen_hyparams[op]['padding'])
            x = tf.nn.relu(x + gen_biases[op])
            print('\t\tX after %s:%s'%(op,x.get_shape().as_list()))

        if op=='loc_res_norm':
            print('\tLocal Response Normalization')
            x = tf.nn.local_response_normalization(x, depth_radius=3, bias=None, alpha=1e-2, beta=0.75)

    return tf.nn.tanh(x)

def calc_generator_loss(disc_out):
    # Training computation.
    loss = -tf.reduce_mean(tf.log(disc_out))
    return loss

def optimize_generator(gen_loss):
    # Optimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5).minimize(gen_loss)
    return optimizer

'''==================================================================
DISCRIMINATOR FUNCTIONS
=================================================================='''
def create_discriminator_layers():
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
                                    stddev=2./min(5,disc_hyparams[op]['weights'][0])
                                    )
            )
            disc_biases[op] = tf.Variable(tf.constant(np.random.random()*0.001,shape=[disc_hyparams[op]['weights'][3]]))

def update_discriminator_fulcon_in(fan_in):
    for op in disc_conv_ops:
        if 'fulcon_out' in op:
            disc_hyparams[op]['in'] = fan_in
            break

def get_discriminator_output(dataset,is_train):

    # Variables.
    if not use_dropout:
        x = dataset
    else:
        x = tf.nn.dropout(dataset,1.0 - in_dropout_rate,seed=tf.set_random_seed(98765))
    print('Calculating inputs for data X(%s)...'%x.get_shape().as_list())
    for op in disc_conv_ops:
        if 'conv' in op:
            print('\tCovolving data (%s)'%op)
            x = tf.nn.conv2d(x, disc_weights[op], disc_hyparams[op]['stride'], padding=disc_hyparams[op]['padding'])
            x = tf.nn.relu(x + disc_biases[op])
            print('\t\tX after %s:%s'%(op,x.get_shape().as_list()))
        if op=='loc_res_norm':
            print('\tLocal Response Normalization')
            x = tf.nn.local_response_normalization(x, depth_radius=3, bias=None, alpha=1e-2, beta=0.75)

        if 'fulcon' in op:
            break

    # we need to reshape the output of last subsampling layer to
    # convert 4D output to a 2D input to the hidden layer
    # e.g subsample layer output [batch_size,width,height,depth] -> [batch_size,width*height*depth]
    shape = x.get_shape().as_list()
    rows = shape[0]
    update_discriminator_fulcon_in(shape[1] * shape[2] * shape[3])

    for op in disc_conv_ops:
        if 'fulcon' in op:
            print('Unwrapping last convolution layer %s to %s hidden layer'%(shape,(rows,disc_hyparams[op]['in'])))
            x = tf.reshape(x, [rows,disc_hyparams[op]['in']])
            break

    for op in disc_conv_ops:
        if 'fulcon_hidden' not in op:
            continue
        else:
            if is_train and use_dropout:
                x = tf.nn.dropout(tf.nn.relu(tf.matmul(x,disc_weights[op])+disc_biases[op]),keep_prob=1.-dropout_rate,seed=tf.set_random_seed(12321))
            else:
                x = tf.nn.relu(tf.matmul(x,disc_weights[op])+disc_biases[op])

    if use_dropout:
        x = tf.nn.dropout(x,1.0-dropout_rate,seed=tf.set_random_seed(98765))

    return tf.nn.sigmoid(tf.matmul(x, disc_weights['fulcon_out']) + disc_biases['fulcon_out'])

def calc_discriminator_loss(disc_out_with_real,disc_out_with_gen):
    # Training computation.
    # e.g. log(0.2) = -0.7, log(1) = 0
    loss = -tf.reduce_mean(tf.log(disc_out_with_real) + tf.log(1.-disc_out_with_gen))
    return loss

def optimize_discriminator(loss):
    # Optimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return optimizer

if __name__=='__main__':

    global train_size,valid_size,test_size
    global log_suffix,data_filename
    global total_iterations

    if dataset_type=='cifar-10':
        (full_train_dataset,full_train_labels),(valid_dataset,valid_labels),(test_dataset,test_labels)=load_data.reformat_data_cifar10(data_filename)

    graph = tf.Graph()

    # Value logger will log info used to calculate policies
    '''test_logger = logging.getLogger('test_logger_'+log_suffix)
    test_logger.setLevel(logging.INFO)
    fileHandler = logging.FileHandler('test_logger_'+log_suffix, mode='w')
    fileHandler.setFormatter(logging.Formatter('%(message)s'))
    test_logger.addHandler(fileHandler)'''

    test_accuracies = []

    with tf.Session(graph=graph) as session:
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
        tf_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

        tf_noise_dataset = tf.placeholder(tf.float32, shape=(batch_size,z_size))

        tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size,image_size,image_size,num_channels))
        tf_test_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

        global_step = tf.Variable(0, trainable=False)
        #start_lr = tf.Variable(start_lr)

        create_generator_layers()
        create_discriminator_layers()

        print('================ Training ==================\n')
        gen_dataset = get_generator_output(tf_noise_dataset)

        disc_gen_out = get_discriminator_output(gen_dataset)
        disc_real_out = get_discriminator_output(tf_dataset)

        gen_loss = calc_generator_loss(disc_gen_out)
        disc_loss = calc_discriminator_loss(disc_real_out,disc_gen_out)

        gen_optimize = optimize_generator(gen_loss)
        disc_optimizer = optimize_discriminator(disc_loss)

        print('==============================================\n')
        tf.initialize_all_variables().run()

        print('Initialized...')
        print('\tBatch size:',batch_size)
        print('\tNum Epochs: ',num_epochs)
        print('\tDropout: ',use_dropout,', ',dropout_rate)
        print('\tDecay step %d'%decay_step)
        print('==================================================\n')

        accuracy_drop = 0 # used for early stopping
        max_test_accuracy = 0

        for epoch in range(num_epochs):



            for iteration in range(floor(float(train_size)/batch_size)):
                offset = iteration * batch_size
                assert offset < train_size
                batch_data = train_dataset[offset:offset + batch_size, :, :, :]
                batch_labels = train_labels[offset:offset + batch_size, :]

                z_data = np.random.uniform(-1.0,1.0,size=[batch_size,z_size]).astype(np.float32)

                gen_feed_dict = {tf_noise_dataset : z_data}
                disc_feed_dict = {tf_dataset : batch_data, tf_noise_dataset : z_data}
                _, gen_images, l, _ = session.run([gen_dataset,disc_gen_out,gen_loss,gen_optimize], feed_dict=gen_feed_dict)
                _, l, _ = session.run([disc_real_out,disc_loss,disc_optimizer], feed_dict=disc_feed_dict)

                if total_iterations % 50 == 0:
                    print('Minibatch loss at epoch,iteration %d,%d: %f' % (epoch,iteration, l))
                    print('Learning rate: %.5f'%updated_lr)
                    print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))

                total_iterations += 1


