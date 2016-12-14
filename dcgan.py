__author__ = 'Thushan Ganegedara'

import tensorflow as tf
import numpy as np


conv_ops = ['fulcon_in','conv_1','pool_1','loc_res_norm','conv_2','pool_2','loc_res_norm','conv_3','pool_1','loc_res_norm']

#number of feature maps for each convolution layer
depth_conv = {'conv_1':128,'conv_2':96,'conv_3':64}

#weights (conv): [width,height,in_depth,out_depth]
#kernel (pool): [_,width,height,_]
conv_1_hyparams = {'weights':[5,5,num_channels,depth_conv['conv_1']],'stride':[1,1,1,1],'padding':'SAME'}
conv_2_hyparams = {'weights':[5,5,depth_conv['conv_1'],depth_conv['conv_2']],'stride':[1,1,1,1],'padding':'SAME'}
conv_3_hyparams = {'weights':[5,5,depth_conv['conv_2'],depth_conv['conv_3']],'stride':[1,1,1,1],'padding':'SAME'}
pool_1_hyparams = {'type':'max','kernel':[1,3,3,1],'stride':[1,3,3,1],'padding':'SAME'}
pool_2_hyparams = {'type':'max','kernel':[1,3,3,1],'stride':[1,2,2,1],'padding':'SAME'}
pool_3_hyparams = {'type':'avg','kernel':[1,3,3,1],'stride':[1,2,2,1],'padding':'SAME'}
#I'm using only one inception module. Hyperparameters for the inception module found here
incept_1_hyparams = {
    'ipool_2x2':{'type':'avg','kernel':[1,5,5,1],'stride':[1,1,1,1],'padding':'SAME'},
    'iconv_1x1':{'weights':[1,1,depth_conv['conv_3'],depth_conv['iconv_1x1']],'stride':[1,1,1,1],'padding':'SAME'},
    'iconv_3x3':{'weights':[3,3,depth_conv['iconv_1x1'],depth_conv['iconv_3x3']],'stride':[1,1,1,1],'padding':'SAME'},
    'iconv_5x5':{'weights':[5,5,depth_conv['iconv_1x1'],depth_conv['iconv_5x5']],'stride':[1,1,1,1],'padding':'SAME'}
}

# fully connected layer hyperparameters
hidden_1_hyparams = {'in':0,'out':1024}
hidden_2_hyparams = {'in':1024,'out':512}
out_hyparams = {'in':124,'out':100}
fulcon_in_hyparams = {'in':100,'out':conv_1_hyparams['weights'][0]*conv_1_hyparams['weights'][1]*conv_1_hyparams['weights'][2]}

hyparams = {'fulcon_in','conv_1': conv_1_hyparams, 'conv_2': conv_2_hyparams, 'conv_3':conv_3_hyparams,'conv_4':conv_4_hyparams,'conv_5':conv_5_hyparams,
           'incept_1': incept_1_hyparams,'pool_1': pool_1_hyparams, 'pool_2':pool_2_hyparams, 'pool_3':pool_3_hyparams,
           'fulcon_hidden_1':hidden_1_hyparams,'fulcon_hidden_2': hidden_2_hyparams, 'fulcon_out':out_hyparams}
