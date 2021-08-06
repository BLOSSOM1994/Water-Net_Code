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
import vgg
import tensorflow.contrib.slim as slim
class T_CNN(object):

  def __init__(self, 
               sess, 
               image_height=230,
               image_width=310,
               label_height=230, 
               label_width=310,
               batch_size=2,
               c_dim=3, 
               checkpoint_dir=None, 
               sample_dir=None
               ):

    self.sess = sess
    self.is_grayscale = (c_dim == 1)
    self.image_height = image_height
    self.image_width = image_width
    self.label_height = label_height
    self.label_width = label_width
    self.batch_size = batch_size
    self.dropout_keep_prob=0.5


    self.c_dim = c_dim
    self.df_dim = 64
    self.checkpoint_dir = checkpoint_dir
    self.sample_dir = sample_dir
    self.vgg_dir='/content/drive/MyDrive/vgg_pretrained/imagenet-vgg-verydeep-19.mat'
    self.CONTENT_LAYER = 'relu5_4'
    self.build_model()

  def build_model(self):
    self.images = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images')
    self.images_wb = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_wb')
    self.images_ce = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_ce')
    self.images_gc = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='images_gc')
    self.labels_image = tf.placeholder(tf.float32, [self.batch_size, self.image_height, self.image_width, self.c_dim], name='labels_image')


    self.images_test = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim], name='images_test')
    self.images_test_wb = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim], name='images_test_wb')
    self.images_test_ce = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim], name='images_test_ce')
    self.images_test_gc = tf.placeholder(tf.float32, [1, self.image_height, self.image_width, self.c_dim], name='images_test_gc')

    self.labels_test = tf.placeholder(tf.float32, [1,self.label_height,self.label_width, self.c_dim], name='labels_test')
    
    self.pred_h1= self.model()

    self.enhanced_texture_vgg1 = vgg.net(self.vgg_dir, vgg.preprocess(self.pred_h1 * 255))
    self.labels_texture_vgg = vgg.net(self.vgg_dir, vgg.preprocess(self.labels_image* 255))
    self.loss_texture1 =tf.reduce_mean(tf.square(self.enhanced_texture_vgg1[self.CONTENT_LAYER]-self.labels_texture_vgg[self.CONTENT_LAYER]))
    
    self.loss_h1= tf.reduce_mean(tf.abs(self.labels_image-self.pred_h1))
    self.loss = 0.05*self.loss_texture1+ self.loss_h1
    t_vars = tf.trainable_variables()

    self.saver = tf.train.Saver(max_to_keep=0)
    
  def train(self, config):
    if config.is_train:     
      data_train_list = prepare_data(self.sess, dataset="/content/drive/MyDrive/waternetDataSets/input_train")
      data_wb_train_list = prepare_data(self.sess, dataset="/content/drive/MyDrive/waternetDataSets/input_wb_train")
      data_ce_train_list = prepare_data(self.sess, dataset="/content/drive/MyDrive/waternetDataSets/input_ce_train")
      data_gc_train_list = prepare_data(self.sess, dataset="/content/drive/MyDrive/waternetDataSets/input_gc_train")
      image_train_list = prepare_data(self.sess, dataset="/content/drive/MyDrive/waternetDataSets/gt_train")

      data_test_list = prepare_data(self.sess, dataset="/content/drive/MyDrive/waternetDataSets/input_test")
      data_wb_test_list = prepare_data(self.sess, dataset="/content/drive/MyDrive/waternetDataSets/input_wb_test")
      data_ce_test_list = prepare_data(self.sess, dataset="/content/drive/MyDrive/waternetDataSets/input_ce_test")
      data_gc_test_list = prepare_data(self.sess, dataset="/content/drive/MyDrive/waternetDataSets/input_gc_test")
      image_test_list = prepare_data(self.sess, dataset="/content/drive/MyDrive/waternetDataSets/gt_test")

      seed = 568
      np.random.seed(seed)
      np.random.shuffle(data_train_list)
      np.random.seed(seed)
      np.random.shuffle(data_wb_train_list)
      np.random.seed(seed)
      np.random.shuffle(data_ce_train_list)
      np.random.seed(seed)
      np.random.shuffle(data_gc_train_list)
      np.random.seed(seed)
      np.random.shuffle(image_train_list)

    else:
      data_test_list = prepare_data(self.sess, dataset="/content/drive/MyDrive/waternetDataSets/input_test")
      data_wb_test_list = prepare_data(self.sess, dataset="/content/drive/MyDrive/waternetDataSets/input_wb_test")
      data_ce_test_list = prepare_data(self.sess, dataset="/content/drive/MyDrive/waternetDataSets/input_ce_test")
      data_gc_test_list = prepare_data(self.sess, dataset="/content/drive/MyDrive/waternetDataSets/input_gc_test")
      image_test_list = prepare_data(self.sess, dataset="/content/drive/MyDrive/waternetDataSets/gt_test")



    sample_data_files = data_test_list[16:20]
    sample_wb_data_files = data_wb_test_list[16:20]
    sample_ce_data_files = data_ce_test_list[16:20]
    sample_gc_data_files = data_gc_test_list[16:20]
    sample_image_files = image_test_list[16:20]

    sample_data = [
          get_image(sample_data_file,
                    is_grayscale=self.is_grayscale) for sample_data_file in sample_data_files]
    sample_lable_image = [
          get_image(sample_image_file,
                    is_grayscale=self.is_grayscale) for sample_image_file in sample_image_files]

    sample_inputs_data = np.array(sample_data).astype(np.float32)
    sample_inputs_lable_image = np.array(sample_lable_image).astype(np.float32)


    self.train_op = tf.train.AdamOptimizer(config.learning_rate,0.9).minimize(self.loss)
    tf.global_variables_initializer().run()
    
    
    counter = 0
    start_time = time.time()

    if self.load(self.checkpoint_dir):
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    if config.is_train:
      print("Training...")
      loss = np.ones(config.epoch)

      for ep in range(config.epoch):
        # Run by batch images
        
        batch_idxs = len(data_train_list) // config.batch_size
        for idx in range(0, batch_idxs):

          batch_files       = data_train_list[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_files_wb       = data_wb_train_list[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_files_ce       = data_ce_train_list[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_files_gc       = data_gc_train_list[idx*config.batch_size:(idx+1)*config.batch_size]
          batch_image_files = image_train_list[idx*config.batch_size : (idx+1)*config.batch_size]


          batch_ = [
          get_image(batch_file,
                    is_grayscale=self.is_grayscale) for batch_file in batch_files]
          batch_wb = [
          get_image(batch_wb_file,
                    is_grayscale=self.is_grayscale) for batch_wb_file in batch_files_wb]
          batch_ce = [
          get_image(batch_ce_file,
                    is_grayscale=self.is_grayscale) for batch_ce_file in batch_files_ce]
          batch_gc = [
          get_image(batch_gc_file,
                    is_grayscale=self.is_grayscale) for batch_gc_file in batch_files_gc]
          batch_labels_image = [
          get_image(batch_image_file,
                    is_grayscale=self.is_grayscale) for batch_image_file in batch_image_files]
          
          batch_input = np.array(batch_).astype(np.float32)
          batch_wb_input = np.array(batch_wb).astype(np.float32)
          batch_ce_input = np.array(batch_ce).astype(np.float32)
          batch_gc_input = np.array(batch_gc).astype(np.float32)
          batch_image_input = np.array(batch_labels_image).astype(np.float32)

          counter += 1
          _, err = self.sess.run([self.train_op, self.loss ], feed_dict={self.images: batch_input, self.images_wb: batch_wb_input, self.images_ce: batch_ce_input, self.images_gc: batch_gc_input, self.labels_image:batch_image_input})
          # print(batch_light)

          if counter % 100 == 0:
            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
              % ((ep+1), counter, time.time()-start_time, err ))
            
          if idx  == batch_idxs-1: 
            batch_test_idxs = len(data_test_list) // config.batch_size
            err_test =  np.ones(batch_test_idxs)
            for idx_test in range(0,batch_test_idxs):

              sample_data_files = data_train_list[idx_test*config.batch_size:(idx_test+1)*config.batch_size]
              sample_wb_files = data_wb_train_list[idx_test*config.batch_size : (idx_test+1)*config.batch_size]
              sample_ce_files = data_ce_train_list[idx_test*config.batch_size : (idx_test+1)*config.batch_size]
              sample_gc_files = data_gc_train_list[idx_test*config.batch_size : (idx_test+1)*config.batch_size]
              sample_image_files = image_train_list[idx_test*config.batch_size : (idx_test+1)*config.batch_size]
             
              sample_data = [get_image(sample_data_file,
                            is_grayscale=self.is_grayscale) for sample_data_file in sample_data_files]
              sample_wb_image = [get_image(sample_wb_file,
                                    is_grayscale=self.is_grayscale) for sample_wb_file in sample_wb_files]
              sample_ce_image = [get_image(sample_ce_file,
                                    is_grayscale=self.is_grayscale) for sample_ce_file in sample_ce_files]
              sample_gc_image = [get_image(sample_gc_file,
                                    is_grayscale=self.is_grayscale) for sample_gc_file in sample_gc_files]

              sample_lable_image = [get_image(sample_image_file,
                                    is_grayscale=self.is_grayscale) for sample_image_file in sample_image_files]

              sample_inputs_data = np.array(sample_data).astype(np.float32)
              sample_inputs_wb_image = np.array(sample_wb_image).astype(np.float32)
              sample_inputs_ce_image = np.array(sample_ce_image).astype(np.float32)
              sample_inputs_gc_image = np.array(sample_gc_image).astype(np.float32)
              sample_inputs_lable_image = np.array(sample_lable_image).astype(np.float32)


              err_test[idx_test] = self.sess.run(self.loss, feed_dict={self.images: sample_inputs_data, self.images_wb: sample_inputs_wb_image, self.images_ce: sample_inputs_ce_image, self.images_gc: sample_inputs_gc_image,self.labels_image:sample_inputs_lable_image})    

            loss[ep]=np.mean(err_test)
            print(loss)
            self.save(config.checkpoint_dir, counter)

            
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
