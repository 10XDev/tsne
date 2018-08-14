#!/usr/bin/env python3

import os
import sys
import gzip
import urllib
import pickle
import matplotlib

import numpy as np

from tsne import bh_sne
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from struct import pack

def get_mnist():

    if not os.path.exists('mnist.pkl.gz'):
        print('downloading MNIST')
        if sys.version_info >= (3, 0):
            urllib.request.urlretrieve(
            'http://deeplearning.net/data/mnist/mnist.pkl.gz', 'mnist.pkl.gz')
        else:
            urllib.urlretrieve(
                        'http://deeplearning.net/data/mnist/mnist.pkl.gz', 'mnist.pkl.gz')
        print('downloaded')

    f = gzip.open("mnist.pkl.gz", "rb")
    if sys.version_info >= (3, 0):
        train, val, test = pickle.load(f, encoding='latin1')
    else:
        train, val, test = pickle.load(f)
    f.close()

    # Get all data in one array
    _train = np.asarray(train[0], dtype=np.float64)
    _val = np.asarray(val[0], dtype=np.float64)
    _test = np.asarray(test[0], dtype=np.float64)
    mnist = np.vstack((_train, _val, _test))

    # Also the classes, for labels in the plot later
    classes = np.hstack((train[1], val[1], test[1]))

    return mnist, classes

def save_samples(samples):

    # Assume that the dimensionality of the first sample is representative for
    #   the whole batch
    sample_dim = len(samples[0])
    sample_count = len(samples)

    # Note: The binary format used by bh_tsne is roughly the same as for
    #   vanilla tsne
    with open(os.path.join(os.path.dirname(__file__), 'data.dat'), 'wb') as data_file:
        # Write the bh_tsne header
        data_file.write(pack('iiddii', sample_count, sample_dim, 0.5, 50, 2, 1000))
        # Then write the data
        for sample in samples:
            data_file.write(pack('{}d'.format(len(sample)), *sample))


if __name__ == '__main__':
    mnist, classes = get_mnist()
    save_samples(mnist)
    X_2d = bh_sne(mnist)

    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y)
    plt.savefig('test-mnist')


