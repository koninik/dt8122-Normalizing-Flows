import matplotlib.pyplot as plt
import torch
import numpy as np

def subfig_plot(position, data, x_start, x_end, y_start, y_end, title, suptitle, color, name, dataset):
    if position == 3:
        plt.clf()
    plt.subplot(2,2,position)
    plt.scatter(data[:, 0], data[:, 1], c=color, s=1)
    #plt.scatter(data, c=color, s=1)
    plt.xlim(x_start,x_end)
    plt.ylim(y_start,y_end)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'./{name}.png')
    plt.suptitle(f'{dataset} - {suptitle}')

def hist_plot(position, data, x_start, x_end, y_start, y_end, title, suptitle, name, dataset):
    plt.subplot(2,2,position)
    #plt.scatter(data[:, 0], data[:, 1], c=color, s=1)
    plt.hist2d(data[:, 0], data[:, 1], bins=300, density=True)#,range=[[x_start, x_end], [y_start, y_end]])
    #plt.xlim(x_start,x_end)
    #plt.ylim(y_start,y_end)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f'./figures/{name}.png')
    plt.suptitle(f'{dataset} - {suptitle}')
    
    
def deriv_tanh(x):
    """ derivative of tanh """
    y = torch.tanh(x)
    return 1.0 - y * y

    

class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()

    def reset(self):
        self.val = None
        self.avg = 0

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val



