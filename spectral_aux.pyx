import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def Mnn(np.ndarray[double, ndim=2] X, int m):
    cdef int n = X.shape[0]
    cdef int dim = X.shape[1]
    cdef int i
    cdef int j
    cdef double s
    cdef int k
    cdef int l
    if n < m:
        raise ValueError
    cdef np.ndarray[int, ndim=2] Mnn = np.empty(shape=(n, m), dtype=np.int)
    cdef np.ndarray[double, ndim=2] dist_S = np.empty(shape=(n, n), dtype=np.double)    # array of distances between points
    cdef np.ndarray[int, ndim=1] indexes = np.empty(shape=m, dtype=np.int)  
    cdef np.ndarray[double, ndim=1] distances = np.empty(shape=m, dtype=np.double)
    for i in range(n):
        for j in range(i+1, n):
            s = 0.0
            for k in range(dim):
                s += (X[i, k]-X[j, k])**2   # distance between i., j. point
            dist_S[i, j] = np.sqrt(s)
            dist_S[j, i] = dist_S[i, j]
        dist_S[i, i] = np.infty
    for i in range(n):
        for j in range(0, m):   # select m nearest neighbours 
            indexes[j] = -1
            distances[j] = np.infty
        for j in range(0, n):
            for k in range(m):
                if dist_S[i, j] < distances[k]:
                    for l in reversed(range(k+1, m)):
                        distances[l] = distances[l-1]
                        indexes[l] = indexes[l-1]
                    distances[k] = dist_S[i, j]
                    indexes[k] = j
                    break
        Mnn[i, :] = indexes
    return Mnn