import os
import time
import model
import sys

warmup = 0
epochs = 12000
batch_sz = 64
op_iter = 30
host = "localhost"
websocket_port = 5858
unix_socket = "./cscl_sock"
pipe_name = "./cscl_pipe"
use_tcp = False
num_workers = 5
gpu = False
send_as_pipe = True
join_work_time = 15
save_iter = 100
dump = open("slurm-reports.log", "a+")

PY_REC = 100
REC_LENGTH = 9
WIDTH = PY_REC * REC_LENGTH
THRESHOLD = 0.7

timestamp = str(int(time.time()))
ckpoint = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))

model = model.Model(WIDTH, 2, [[1,data_helpers.REC_LENGTH,1,32], [20,REC_LENGTH,32,64]], [900, 1800, 900], PY_REC, REC_LENGTH, gpu=gpu, dump=dump)
dump.flush()