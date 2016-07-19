from __future__ import print_function
import tensorflow as tf
import numpy as np
import simplejson as json
import base64

class ParameterServerModel():

   def __init__(self, x, y_, keep_prob, dropout, compute_gradients, apply_gradients, minimize, accuracy, session, batch_size, out_final, gpu, dump):
      self.session = session
      self.batch_size = batch_size
      self.graph = session.graph
      self.session.graph.as_default().__enter__()
      self.dump = dump
      self.x = x
      self.y_ = y_
      self.keep_prob = keep_prob
      self.dropout = dropout
      self.out_final = out_final
      self.gpu = gpu
      self.compute_gradients = compute_gradients
      self.apply_gradients = apply_gradients
      self.acc = accuracy
      self.minimize = minimize
      self.reset_gradients()
      self.gradient_counter = tf.Variable(initial_value=0, trainable=False)

      with tf.device('/gpu:0' if self.gpu else '/cpu:0'):
         self.parameter_assignments = [None]*len(self.compute_gradients)
         for i in xrange(len(self.compute_gradients)):
             gradient = self.compute_gradients[i][0]
             variable = self.compute_gradients[i][1]
             if gradient is not None:
                self.parameter_assignments[i] = variable.assign(gradient)
             else:
                print("Gradient was None for variable with specified index! (" + str(i) + ")", file=self.dump)
         self.session.run(tf.initialize_all_variables())

   def get_num_classes(self):                  
      return self.y_.get_shape().as_list()[1]
      
   def train(self, labels, features):
      with self.session.as_default():
         with tf.device('/gpu:0' if self.gpu else '/cpu:0'):
            feed = {self.x: features, self.y_: labels, self.keep_prob: self.dropout}
            for i in range(len(self.compute_gradients)):
               if self.compute_gradients[i][0] is not None:
                  self.gradients[i] += self.compute_gradients[i][0].eval(feed_dict=feed)
                  self.num_gradients += 1
               else:
                  print("Gradient was None! Hope it is the non trainable one from above: " + str(i), file=self.dump)
            del feed
         
   def test(self, labels, features):
      with self.session.as_default():
         with tf.device('/gpu:0' if self.gpu else '/cpu:0'):
            feed = {self.x: features, self.y_: labels, self.keep_prob: 1.0}
            test_error_rate = self.acc.eval(feed_dict=feed)
            del feed
            return test_error_rate
         
   def get_parameters(self):
      with self.session.as_default():
         with tf.device('/gpu:0' if self.gpu else '/cpu:0'):
            result = [None]*len(self.compute_gradients)
            for i in xrange(len(self.compute_gradients)):
               result[i] = self.compute_gradients[i][1].eval(session=self.session)
            array = np.array(result)
            del result[:]
            del result
            return array
         
   def assign_parameters(self, parameters):
      with self.session.as_default():
         with tf.device('/gpu:0' if self.gpu else '/cpu:0'):
            self.reset_gradients()
            for i, grad_var in enumerate(self.compute_gradients):
               if self.parameter_assignments[i] is not None:
                  self.parameter_assignments[i].eval(feed_dict={grad_var[0]:parameters[i]})
               else:
                  print("Gradient was None! Hope it is the non trainable one from above: " + str(i), file=self.dump)
            
   def apply(self, gradients):
      with self.graph.as_default():
         with tf.device('/gpu:0' if self.gpu else '/cpu:0'):
            feed_dict = {}
            i = 0
            for grad_var in self.compute_gradients:
               if grad_var[0] is not None:
                  feed_dict[grad_var[0]] = gradients[i]
               i += 1
            self.apply_gradients.run(session=self.session, feed_dict=feed_dict)
            del feed_dict
            del gradients
         
   def get_gradients(self):
      result = [None]*len(self.gradients)
      for i in xrange(len(self.gradients)):
         result[i] = np.divide(self.gradients[i], self.num_gradients).astype('float32')
      array = np.array(result)
      del result[:]
      del result
      return array
      
   def reset_gradients(self):
      with self.session.as_default():
         with tf.device('/gpu:0' if self.gpu else '/cpu:0'):
            self.gradients = [tf.zeros(g[1].get_shape()).eval() for g in self.compute_gradients]
            self.num_gradients = 0
         
   def train_warmup(self, partition):
      error_rates = []
      iteration = 0
      batch_size = self.batch_size
      for i in range(0, len(partition), batch_size):
         data = partition[i:i+batch_size]
         labels, features = self.process_data(data)
         if len(labels) is 0:
            break
         with self.session.as_default():
            with tf.device('/gpu:0' if self.gpu else '/cpu:0'):
               feed = {self.x: features, self.y_: labels, self.keep_prob: self.dropout}
               self.minimize.run(feed_dict = feed)
               error_rate = self.acc.eval(feed_dict=feed)
               iteration += 1
               print("Warmup training iteration " + str(iteration) + " at " + str(error_rate) + " accuracy", file=self.dump)
      return error_rates
      
   def process_data(self, data):
      raise AssertionError('function not implemented')
      
   def process_partition(self, partition):
      raise AssertionError('function not implemented')
      
   def saveWith(self, saver, where, gi):
      saver.save(self.session, where, global_step=gi)
      
   def serialize(self, array):
      return array.dumps()
      
   def deserialize(self, serialized):
      return np.loads(serialized)
