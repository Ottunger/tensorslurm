#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import tensorflow as tf
import math
import numpy as np
import data_helpers
import time
import sys
import socket
import constants
import os

warmup = constants.warmup
epochs = constants.epochs
batch_sz = constants.batch_sz
op_iter = constants.op_iter
host = constants.host
websocket_port = constants.websocket_port
unix_socket = constants.unix_socket
use_tcp = constants.use_tcp
num_workers = constants.num_workers
ckpoint = constants.ckpoint
model = constants.model
send_as_pipe = constants.send_as_pipe
pipe_name = constants.pipe_name
join_work_time = constants.join_work_time
save_iter = constants.save_iter
dump = constants.dump

class TensorSlurmWorker:

   def __init__(self, batch_size, websocket_port, windex):
      print("[%s] Creating new worker" % windex, file=dump)
      dump.flush()
      self.model = model
      self.batch_size = batch_size
      self.websocket_port = websocket_port
      self.iteration = 1
      self.windex = str(windex)
      self.make_connection(req=True, join=True)
      
   def make_connection(self, req, join=False):
      if send_as_pipe:
         if not os.path.exists(pipe_name + self.windex):
            os.mkfifo(pipe_name + self.windex)
         self.pipeout = os.open(pipe_name + self.windex, os.O_WRONLY)
         if join:
            time.sleep(join_work_time)
      try:
         if use_tcp:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.connect((host, websocket_port))
         else:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.connect(unix_socket)
         self.wfile = self.sock.makefile('wb', 0)
         if req:
            self.sock.sendall("connect")
            self.request_parameters()
      except:
         raise
      
   def train_partition(self, partition):
      labels, features = self.model.process_partition(partition)
      np.random.seed(int(time.time()))
      while True:
         shuffle_indices = np.random.permutation(np.arange(len(labels)))
         x_shuffled = []
         y_shuffled = []
         for i in xrange(min(len(shuffle_indices), self.batch_size)):
            x_shuffled.append(features[shuffle_indices[i]])
            y_shuffled.append(labels[shuffle_indices[i]])
         if len(labels) is 0:
            print("[%s] This worker is out of work, shutting down" % self.windex, file=dump)
            dump.flush()
            break
         if self.time_to_pull(self.iteration):
            self.make_connection(req=True)
         self.model.train(y_shuffled, x_shuffled)
         self.iteration += 1
         print("[%s] Just completed an iteration, chief!" % self.windex, file=dump)
         dump.flush()
         if self.time_to_push(self.iteration):
            self.push_gradients()
      
   def test_partition(self, partition):
      labels, features = self.model.process_partition(partition)
      self.request_parameters()
      error_rate = self.model.test(labels, features)
      return [error_rate]
      
   def test(self, data):
      if len(data) is 0:
         return 1.0
      self.request_parameters()
      accuracy = self.model.test(data)
      return accuracy
      
   def time_to_pull(self, iteration):
      return iteration % op_iter == 0
      
   def time_to_push(self, iteration):
      return iteration % op_iter == 0
      
   def request_parameters(self):
      print("[%s] Read at socket..." % self.windex, file=dump)
      parameters = ""
      i = 0
      #Wait for the server to have processed data
      while True:
         rcv = self.sock.recv(4096)
         if rcv.startswith("ok"):
            parameters = rcv.lstrip("ok")
            print("[%s] Server starts sending..." % self.windex, file=dump)
            break
      while True:
         rcv = self.sock.recv(4096)
         if rcv.endswith("end"):
            parameters += rcv.rstrip("end")
            break
         parameters += rcv
         i = i+1
         if i%9600 == 0:
            print("Read already %d..." % sys.getsizeof(parameters), file=dump)
      print("[%s] Installing up-to-date gradients..." % self.windex, file=dump)
      parameters = self.model.deserialize(parameters)
      self.model.assign_parameters(parameters)
      print("[%s] Installed up-to-date gradients..." % self.windex, file=dump)
      
   def push_gradients(self):
      print("[%s] Prepare our data..." % self.windex, file=dump)
      gradients = self.model.get_gradients()
      serialized = self.model.serialize(gradients)
      del gradients
      print("[%s] Sending %d bytes of data..." % (self.windex, sys.getsizeof(serialized)), file=dump)
      self.make_connection(req=False)
      try:
         if send_as_pipe:
            self.wfile.write("once" + pipe_name + self.windex)
            self.wfile.flush()
            os.write(self.pipeout, serialized.encode('string_escape') + "\n")
         else:
            self.wfile.write("once" + serialized + "end")
            self.wfile.flush()
         print("[%s] Written and sent..." % self.windex, file=dump)
         dump.flush()
      except Exception,e:
         print("[%s] Error sending. Attempt to remake connection..." % self.windex, file=dump)
         print(e, file=dump)
         self.iteration += 1
         self.make_connection(req=True)
         
   def wait_msg(self):
      rcv = ""
      while True:
         rcv = self.sock.recv(10)
         if rcv is not "" and rcv is not None:
            break
      
if __name__ == "__main__":
   # TODO: Gather test data not from end of files which are always positive instances.
   worker_index = int(sys.argv[len(sys.argv) - 1])
   tr_data = data_helpers.load_data(sys.argv[1:len(sys.argv)-1])
   psize = len(tr_data)//num_workers
   tr_data_split = tr_data[psize*worker_index:psize*worker_index+psize]
   del tr_data
   client = TensorSlurmWorker(batch_sz, websocket_port, worker_index)
   client.train_partition(tr_data_split)
