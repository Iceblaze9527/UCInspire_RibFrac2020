import sys
import time
import datetime

import torch
from torch.nn import DataParallel
import matplotlib.pyplot as plt

##TODO(3) logger module
class Logger(object):
    def __init__(self,log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w")

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

    
def gpu_manager(model):
    device_cnt = torch.cuda.device_count()
    if device_cnt > 0:
        if device_cnt == 1:
            print('Only 1 GPU is available.')
        else:
            print(f"{device_cnt} GPUs are available.")
            model = DataParallel(model)
        model = model.cuda()
    else:
        print('Only CPU is available.')
        
    return model


def draw_curve(data, graph_path=None):
    x, y, _ = data
    fig, ax = plt.subplots()
    ax.plot(y, x)

    if graph_path is not None:
        fig.savefig(graph_path)
    
    plt.close(fig)
    
    return fig


timestamp = lambda: time.asctime(time.localtime(time.time()))
tic = lambda: time.time()
delta_time = lambda start, end: str(datetime.timedelta(seconds=round(end-start,3)))
