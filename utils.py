import sys
import time
import datetime

import torch

##TODO(2)
class Logger(object):
    def __init__(self,log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def set_global_seed(seed=15):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False

timestamp = lambda: time.asctime(time.localtime(time.time()))
tic = lambda: time.time()
delta_time = lambda start, end: str(datetime.timedelta(seconds=round(end-start,3)))