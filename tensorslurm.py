from __future__ import print_function
import threading
import SocketServer as ss
import tensorflow as tf
import numpy as np
import sys
import os
import socket
import data_helpers
import constants

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

class ParameterServerWebsocketHandler(ss.StreamRequestHandler):
   def handle(self):
      """
      Server each request independantly
      while True:
      """
      mode = self.request.recv(4096)
      if mode.startswith("connect"):
         print("A new request for params has arrived!", file=dump)
         self.send_parameters()
      if mode.startswith("once"):
         self.message = mode.lstrip("once")
         print("A new report has arrived from %s!" % self.message, file=dump)
         dump.flush()
         i = 0
         if send_as_pipe:
            self.message = self.server.pipein[self.message].readline()[:-1]
            self.message = self.message.decode('string_escape')
            print("Read already %d..." % sys.getsizeof(self.message), file=dump)
         else:
            while True:
               rcv = self.request.recv(4096)
               if rcv.endswith("end"):
                  self.message += rcv.rstrip("end")
                  break
               self.message += rcv
               i = i+1
               if i%9600 == 0:
                  print("Read already %d..." % sys.getsizeof(self.message), file=dump)
         read_grd = self.server.model.deserialize(self.message)
         self.server.gradient_count += 1
         self.server.lock.acquire()
         self.server.model.apply(read_grd)
         print("Gradient applied for a count up to %d" % self.server.gradient_count, file=dump)
         dump.flush()
         if self.server.gradient_count % save_iter == 0:
            acc = self.server.model.test(self.server.test_labels, self.server.test_features)
            print("Gradients received: %d, accuracy on validation (%d labels): %f" % (self.server.gradient_count, len(self.server.test_labels), acc), file=dump)
            self.server.model.saveWith(self.server.saver, self.server.checkpoint_prefix, self.server.gradient_count)
         self.server.lock.release()
         del read_grd
      
         if(self.server.gradient_count >= self.server.num_epochs):
            self.server.shutdown()
      
   def send_parameters(self):
      self.server.lock.acquire()
      parameters = self.server.model.get_parameters()
      self.server.lock.release()
      serialized = self.server.model.serialize(parameters)
      self.wfile.write("ok" + serialized + "end")
      self.wfile.flush()
      print("Sent the parameters to a friend", file=dump)
      dump.flush()
      
class TCPUnixServer(ss.TCPServer):
   address_family = socket.AF_UNIX
   
   def __init__(self, addr, handler):
      try:
         os.unlink(addr)
      except OSError:
         if os.path.exists(addr):
             raise
      ss.TCPServer.__init__(self, addr, handler)
      
def start_parameter_server(model, num_epochs, warmup_data=None, test_data=None):
   if use_tcp:
      server = ss.TCPServer(unix_socket, ParameterServerWebsocketHandler)
   else:
      server = TCPUnixServer(unix_socket, ParameterServerWebsocketHandler)
   server.model = model
   # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
   checkpoint_dir = os.path.abspath(os.path.join(ckpoint, "checkpoints"))
   server.checkpoint_prefix = os.path.join(checkpoint_dir, "model")
   if not os.path.exists(checkpoint_dir):
       os.makedirs(checkpoint_dir)
   server.saver = tf.train.Saver()
   server.lock = threading.Lock()
   warmup(warmup_data)
   server.gradient_count = 1
   server.num_epochs = num_epochs
   server.pipein = {}
   for i in xrange(num_workers):
      if not os.path.exists(pipe_name + str(i)):
         os.mkfifo(pipe_name + str(i))
      server.pipein[pipe_name + str(i)] = open(pipe_name + str(i), "r")
   server.test_labels, server.test_features = model.process_data(test_data)
   print("Listening if can reach workers and start sending work", file=dump)
   dump.flush()
   server.serve_forever()
      
def warmup(model, data=None):
   if data is not None:
      model.train_warmup(partition=data, error_rates_filename=error_rates_path)
   
def main(warmup_iterations, num_epochs, files, testfile):
   tr_data = data_helpers.load_data(files)
   test_data = data_helpers.load_data([testfile])
   warmup_data = None
   start_parameter_server(model=model, num_epochs=num_epochs, warmup_data=warmup_data, test_data=test_data)

if __name__ == "__main__":        
   main(warmup_iterations=warmup, num_epochs=epochs, files=sys.argv[1:len(sys.argv)-3], testfile=sys.argv[len(sys.argv)-1])
