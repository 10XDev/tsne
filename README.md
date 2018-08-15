# Python-TSNE

This is based on the [10XDev/tsne](https://github.com/10XDev/tsne.git) fork of L.J.P. van der Maaten BH-tSNE implementation.

It has fixes to allow this to run in Python 3 and performance has been significantly
increased with OpenMP parallelism. (see: [tsne-perf-test](https://github.com/rappdw/tsne-perf-test.git))

**Note:** While [Scikit-learn v0.17](http://scikit-learn.org/stable/whats_new.html#version-0-17)
This implementation performs significantly faster than scikit-learn's. If you need speed, use this.



## Algorithms

### Barnes-Hut-SNE

A python ([cython](http://www.cython.org)) wrapper for [Barnes-Hut-SNE](http://homepage.tudelft.nl/19j49/t-SNE.html) aka fast-tsne.

We branched 10XDev's implementation and openmp enabled the code.

## Installation

### Pip

This library has been added to pypi as tsne-mp
You can install the package from the Github repository:

```
pip install git+https://github.com/rappdw/tsne.git
```

Or using docker (to build a wheel for linux):

```
$ bin/build-image jenkins
$ bin/run-image jenkins
$ (in container) python setup.py bdist_wheel

# This will generate a wheel in your project's dist directory
```
### OSX

First install gcc (for openmp support) via: `brew install gcc --without-multilib`. Then
install [CMake](https://cmake.org/). Once that's in place you can run pip or setup.py 
to build a wheel or install locally, e.g.
* `export PATH="/Applications/CMake.app/Contents/bin/:$PATH"; export CC=/usr/local/bin/gcc-6; export CXX=/usr/local/bin/g++-6; python setup.py bdist_wheel`
* `export PATH="/Applications/CMake.app/Contents/bin/:$PATH"; export CXX=/usr/local/bin/g++-6; export CC=/usr/local/bin/gcc-6; pip install -e .`

## Usage

Basic usage:

```
from tsne import bh_sne
X_2d = bh_sne(X)
```
Or, the wheels also contain an executable that can be used from the command-line as described
in [the original project](https://github.com/lvdmaaten/bhtsne).

### Examples

* [Iris](http://nbviewer.ipython.org/urls/raw.github.com/danielfrg/py_tsne/master/examples/iris.ipynb)
* [MNIST](http://nbviewer.ipython.org/urls/raw.github.com/danielfrg/py_tsne/master/examples/mnist.ipynb)
* [word2vec on presidential speeches](https://github.com/prateekpg2455/U.S-Presidential-Speeches) via [@prateekpg2455](https://github.com/prateekpg2455)

## More Information

See *Barnes-Hut-SNE* (2013), L.J.P. van der Maaten. It is available on [arxiv](http://arxiv.org/abs/1301.3342).

## Mulit-core
see: [Python 3 - Multicore](http://python-notes.curiousefficiency.org/en/latest/python3/multicore_python.html)

## Performance

This fork of the orignal project has a number of performance improvements resulting in an order
of magnitude performance improvement when running on multi-core systems. See 
[tsne-pref-test](https://github.com/rappdw/tsne-perf-test) for performance comparisions of
various implementaitons of tsne. 

