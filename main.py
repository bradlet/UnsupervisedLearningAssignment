# Bradley Thompson
# Programming Assignment #3
# CS 545 - Anthony Rhodes

import os
import numpy as np

def load_data(name):
    return np.loadtxt(fname=name, delimiter=",", dtype=np.dtype(np.uint8))

if __name__ == '__main__':
    os.chdir('./dataset')
