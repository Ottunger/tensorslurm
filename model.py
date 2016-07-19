import tensorflow as tf
import numpy as numpy
import sys
from parameterservermodel import ParameterServerModel

def weight_variable(shape, name=None):
   initial = tf.truncated_normal(shape, stddev=0.1)
   return tf.Variable(initial, name=name)

def bias_variable(shape, name=None):
   initial = tf.constant(0.1, shape=shape)
   return tf.Variable(initial, name=name)

class Model(ParameterServerModel):

   def __init__(self, input_size, output_size, conv_filter_sizes, dense_layer_sizes, image_width,
                 image_height, dropout = 0.5, learning_rate = 1e-4, channels = 1, l2norm = 1e-5, gpu=False,
                 recall = 'input_x', dropname = 'drop_pb', predict = 'predictions', batch_size=64, dump=sys.stdout):
      session = tf.InteractiveSession()
      
      training_accuracies = []
      validation_accuracies = []
      training_steps = []
      features_weights = []
      test_accuracy = 0
      
      x = tf.placeholder('float', shape = [None, image_width, image_height, channels], name=recall)
      y_ = tf.placeholder('float', shape = [None, output_size])
      keep_prob = tf.placeholder('float', name=dropname)
      
      with tf.device('/gpu:0' if gpu else '/cpu:0'):
         # Build the network structure
         previous_inputs_size = input_size
         inputs_next = x
         for conv_layer in conv_filter_sizes:
            cur_filter =  weight_variable(conv_layer)
            cur_conv_bias = bias_variable([conv_layer[-1]])
            out_conv = tf.nn.conv2d(inputs_next, cur_filter, strides=[1,1,1,1], padding='SAME')
            inputs_next = tf.nn.relu(out_conv + cur_conv_bias)

         new_width = out_conv.get_shape()[1]
         new_height = out_conv.get_shape()[2]
         new_channels = out_conv.get_shape()[3]
         out_pool_flat = tf.reshape(out_conv, [-1, int(new_channels*new_height*new_width)])

         # build some densely connected layers
         inputs_next = out_pool_flat
         previous_inputs_size = int(new_height*new_width*new_channels)
         for layer_size in dense_layer_sizes[1:-1]:
            cur_weights = weight_variable([previous_inputs_size, layer_size])
            cur_bias = bias_variable([layer_size])
            cur_neurons = tf.nn.relu(tf.matmul(inputs_next, cur_weights) + cur_bias)
            cur_dropout = tf.nn.dropout(cur_neurons, keep_prob = keep_prob)
            inputs_next = cur_dropout
            previous_inputs_size = layer_size

         final_inputs = inputs_next
         prev_add_size = 0

         # This is the last layer
         out_weights = weight_variable([previous_inputs_size + prev_add_size, output_size])
         out_bias = bias_variable([output_size])
         out_final = tf.nn.softmax(tf.matmul(final_inputs, out_weights) + out_bias, name=predict)
         # define cost function
         cost_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(out_final,1e-10,1.0))) + tf.reduce_sum(tf.clip_by_value(l2norm*cur_weights,1e-10,1.0))

         # define optimiser
         opt = tf.train.AdamOptimizer(learning_rate)
         compute_gradients = opt.compute_gradients(cost_entropy, tf.trainable_variables())
         apply_gradients = opt.apply_gradients(compute_gradients)
         minimize = opt.minimize(cost_entropy)

         # define accuracy
         correct_prediction = tf.equal(tf.argmax(out_final,1), tf.argmax(y_,1))
         accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

      ParameterServerModel.__init__(self, x, y_, keep_prob, dropout, compute_gradients, apply_gradients, minimize, accuracy, session, batch_size, out_final, gpu, dump)
      
   def process_data(self, data):
      labels = []
      features = []
      for rec in data:
         if rec[0] > 0.7:
            labels.append([0, 1])
         else:
            labels.append([1, 0])
         features.append(numpy.expand_dims(numpy.reshape(rec[1:], (100, 9)), axis=2))
      return labels, features
      
   def process_partition(self, partition):
      return self.process_data(partition)
