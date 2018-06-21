# distutils: language = c++
import numpy as np
cimport numpy as np
cimport cython
from libcpp cimport bool

cdef extern from "tsne.h" namespace "TSNE":
    void c_run "TSNE::run" (double* X, int N, int D, double* Y, int no_dims,
                            double perplexity, double theta,
                            int rand_seed, bool skip_random_init,
                            double *init, bool use_init,
                            int max_iter, int stop_lying_iter, int mom_switch_iter) nogil

cdef class BH_SNE:
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @staticmethod
    def run(X, int N, int D, int d,
            double perplexity, double theta,
            int seed, init, bool use_init,
            int max_iter, int stop_lying_iter, int mom_switch_iter):
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] _X = np.ascontiguousarray(
                X,
                dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] _init = np.ascontiguousarray(
                init,
                dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] Y = np.zeros(
            (N, d),
            dtype=np.float64,
            order='C')
        assert(N, X.shape[1])
        with nogil:
            c_run(&_X[0,0], N, D, &Y[0,0], d,
                  perplexity, theta,
                  seed, False, &_init[0,0], use_init,
                  max_iter, stop_lying_iter, mom_switch_iter)
        return Y
