# Bradley Thompson
# Programming Assignment #3
# CS 545 - Anthony Rhodes

import os
import numpy as np


def load_data(name):
    return np.loadtxt(fname=name, dtype=np.dtype(np.float), usecols=range(2))


if __name__ == '__main__':
    os.chdir('./dataset')
    dataset = load_data('545_cluster_dataset.txt')

    print(dataset[0])
