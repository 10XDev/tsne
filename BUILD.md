# Building & Distributing tsne-hp
tsne-hp is a binary only distribution. While we are working on converting this library to 
travis-ci, currently we are using a docker based build for linux and platform native build for
OSX and Windows.

## Build Prerequisites

* [numpy](numpy.scipy.org)
* [scipy](http://www.scipy.org/)
* [cython](cython.org)
* [cblas](http://www.netlib.org/blas/) or [openblas](https://github.com/xianyi/OpenBLAS). Tested version is v0.2.5 and v0.2.6 (not necessary for OSX).
* [OpenMp](http://www.openmp.org/)


## Linux

Run `bin/build-linux.sh`

## OSX

## Windows

Nothing Yet



