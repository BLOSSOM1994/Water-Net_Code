from utils import ( 
  imsave,
  prepare_data
)

import time
import os
import matplotlib.pyplot as plt
import re
import numpy as np
import tensorflow.compat.v1 as tf
import scipy.io as scio
from ops import *

class T_CNN(object):

  def __init__(self, 
               sess, 
               image_height=460,
               image_width=620,
               label_height=460, 
               label_width=620,
               batch_size=1,
               c_dim=3, 
               c_depth_dim=1,
               checkpoint_dir=None, 
               sample_dir=None,
               test_image_name = None,
               test_wb_name = None,
               test_ce_name = None,
               test_gc_name = None,
               id = None
               ):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_height = image_height
    self.image_width = image_width
    self.label_height = label_height
    self.label_width = label_width
    self.batch_size = batch_size
    self.dropout_keep_prob=0.5
    self.test_image_name = test_image_name
    self.test_wb_name = test_wb_name
    self.test_ce_name = test_ce_name
    self.test_gc_name = test_gc_name
    self.id = id
    self.c_dim = c_dim
    self.df_dim = 64
    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    #self.c_depth_dim=c_depth_dim
    #
    # self.d_bn1 = batch_norm(name='d_bn1')
    # self.d_bn2 = batch_norm(name='d_bn2')
    # self.d_bn3 = batch_norm(name='d_bn3')
    self.build_model()

  def build_model(self):
    self.images = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images')
    self.images_wb = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_wb')
    self.images_ce = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_ce')
    self.images_gc = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_gc')
    self.pred_h = self.model()


    self.saver = tf.train.Saver()
     
  def train(self, config):


    # Stochastic gradient descent with the standard backpropagation,var_list=self.model_c_vars
    image_test =  get_image(self.test_image_name,is_grayscale=False)
    shape = image_test.shape
    expand_test = image_test[np.newaxis,:,:,:]
    expand_zero = np.zeros([self.batch_size-1,shape[0],shape[1],shape[2]])
    batch_test_image = np.append(expand_test,expand_zero,axis = 0)

    wb_test =  get_image(self.test_wb_name,is_grayscale=False)
    shape = wb_test.shape
    expand_test = wb_test[np.newaxis,:,:,:]
    expand_zero = np.zeros([self.batch_size-1,shape[0],shape[1],shape[2]])
    batch_test_wb = np.append(expand_test,expand_zero,axis = 0)

    ce_test =  get_image(self.test_wb_name,is_grayscale=False)
    shape = ce_test.shape
    expand_test = ce_test[np.newaxis,:,:,:]
    expand_zero = np.zeros([self.batch_size-1,shape[0],shape[1],shape[2]])
    batch_test_ce = np.append(expand_test,expand_zero,axis = 0)

    gc_test =  get_image(self.test_wb_name,is_grayscale=False)
    shape = gc_test.shape
    expand_test = gc_test[np.newaxis,:,:,:]
    expand_zero = np.zeros([self.batch_size-1,shape[0],shape[1],shape[2]])
    batch_test_gc = np.append(expand_test,expand_zero,axis = 0)


    tf.global_variables_initializer().run()
    
    
    counter = 0
    start_time = time.time()

    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")
    start_time = time.time()   
    result_h  = self.sess.run(self.pred_h, feed_dict={self.images: batch_test_image,self.images_wb: batch_test_wb,self.images_ce: batch_test_ce,self.images_gc: batch_test_gc})
    all_time = time.time()
    final_time=all_time - start_time
    print(final_time)


    _,h ,w , c = result_h.shape
    for id in range(0,1):
        result_h0 = result_h[id,:,:,:].reshape(h , w , 3)
        result_h0 = result_h0.squeeze()
        image_path0 = os.path.join(os.getcwd(), config.sample_dir)
        image_path = os.path.join(image_path0, self.test_image_name)
        imsave_lable(result_h0, image_path)

           
    def lrelu(x):
        return tf.maximum(x*0.2,x)

  def upsample_and_concat(x1, x2, output_channels, in_channels):

        pool_size = 2
        deconv_filter = tf.Variable(tf.truncated_normal( [pool_size, pool_size, output_channels, in_channels], stddev=0.02))
        deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1] )

        deconv_output =  tf.concat([deconv, x2],3)
        deconv_output.set_shape([None, None, None, output_channels*2])

        return deconv_output
            

      
  def model(self):

    with tf.variable_scope("main_branch") as scope3: 
#Unet
        conb0 = tf.concat(axis = 3, values = [self.images,self.images_wb,self.images_ce,self.images_gc])
        conv1=slim.conv2d(conb0,16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_1')
        conv1=slim.conv2d(conv1,16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_2')
        pool1=slim.conv2d(conv1,16,[3,3], stride=2, rate=1, activation_fn=lrelu, scope='pooling1' )

        conv2=slim.conv2d(pool1,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_1')
        conv2=slim.conv2d(conv2,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_2')
        pool2=slim.conv2d(conv2,32,[3,3], stride=2, rate=1, activation_fn=lrelu, scope='pooling2' )

        conv3=slim.conv2d(pool2,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_1')
        conv3=slim.conv2d(conv3,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_2')
        pool3=slim.conv2d(conv3,64,[3,3], stride=2, rate=1, activation_fn=lrelu, scope='pooling3' )


        conv4=slim.conv2d(pool3,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv4_1')
        conv4=slim.conv2d(conv4,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv4_2')
        pool4=slim.conv2d(conv4,128,[3,3], stride=2, rate=1, activation_fn=lrelu, scope='pooling4' )


        conv5=slim.conv2d(pool4,256,[3,3], rate=1, activation_fn=lrelu,scope='g_conv5_1')
        conv_global = tf.reduce_mean(conv5,axis=[1,2])
        conv_dense = tf.layers.dense(conv_global,units=128,activation=tf.nn.relu)
        feature = tf.expand_dims(conv_dense,axis=1)
        feature = tf.expand_dims(feature,axis=2)
        ones = tf.zeros(shape=tf.shape(conv4))
        global_feature = feature + ones

        up6 =  tf.concat([conv4, global_feature], axis=3)
        conv6=slim.conv2d(up6,  128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_1')
        conv6=slim.conv2d(conv6,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_2')

        up7 =  upsample_and_concat( conv6, conv3, 64, 128  )
        conv7=slim.conv2d(up7,  64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_1')
        conv7=slim.conv2d(conv7,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_2')

        up8 =  upsample_and_concat( conv7, conv2, 32, 64 )
        conv8=slim.conv2d(up8,  32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_1')
        conv8=slim.conv2d(conv8,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_2')

        up9 =  upsample_and_concat( conv8, conv1, 16, 32 )
        conv9=slim.conv2d(up9,  16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_1')
        conv9=slim.conv2d(conv9,16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_2')

        conv9 = conb0 * conv9
        deconv_filter = tf.Variable(tf.truncated_normal([2, 2, 3, 16], stddev=0.02))
        conv10 = tf.nn.conv2d_transpose(conv9, deconv_filter, tf.shape(input), strides=[1, 2, 2, 1])
        conv_wb77 = slim.conv2d(conv10, 3, [3, 3],rate=1,activation_fn=nn.tanh,scope='out') * 0.58 + 0.52
  #TFU Row+Wb      
        conb00 = tf.concat(axis = 3, values = [self.images,self.images_wb]) 
        conv_wb9 = tf.nn.relu(conv2d(conb00, 3,32, k_h=7, k_w=7, d_h=1, d_w=1,name="conv2wb_9"))
        conv_wb10 = tf.nn.relu(conv2d(conv_wb9, 32,32, k_h=5, k_w=5, d_h=1, d_w=1,name="conv2wb_10"))
        wb1 =tf.nn.relu(conv2d(conv_wb10, 32,3, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2wb_11"))
  #TFU Row+Ce
        conb11 = tf.concat(axis = 3, values = [self.images,self.images_ce]) 
        conv_wb99 = tf.nn.relu(conv2d(conb11, 3,32, k_h=7, k_w=7, d_h=1, d_w=1,name="conv2wb_99"))
        conv_wb100 = tf.nn.relu(conv2d(conv_wb99, 32,32, k_h=5, k_w=5, d_h=1, d_w=1,name="conv2wb_100"))
        ce1 =tf.nn.relu(conv2d(conv_wb100, 32,3, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2wb_111"))
  #TFU Row+Gc
        conb111 = tf.concat(axis = 3, values = [self.images,self.images_gc]) 
        conv_wb999 = tf.nn.relu(conv2d(conb111, 3,32, k_h=7, k_w=7, d_h=1, d_w=1,name="conv2wb_999"))
        conv_wb1000 = tf.nn.relu(conv2d(conv_wb999, 32,32, k_h=5, k_w=5, d_h=1, d_w=1,name="conv2wb_1000"))
        gc1 =tf.nn.relu(conv2d(conv_wb1000, 32,3, k_h=3, k_w=3, d_h=1, d_w=1,name="conv2wb_1111"))
  #Output
        weight_wb,weight_ce,weight_gc=tf.split(conv_wb77,3,3)
        output1=tf.add(tf.add(tf.multiply(wb1,weight_wb),tf.multiply(ce1,weight_ce)),tf.multiply(gc1,weight_gc))


    return output1
 



  def save(self, checkpoint_dir, step):
    model_name = "coarse.model"
    model_dir = "%s_%s" % ("coarse", self.label_height)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir) 

    self.saver.save(self.sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, checkpoint_dir):
    print(" [*] Reading checkpoints...")
    model_dir = "%s_%s" % ("coarse", self.label_height)
    checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
        return True
    else:
        return False
