cimport cython


# --------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline double dot(double *a, double *b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
cdef inline void cross(double *a, double *b, double *v):
    v[0] = a[1]*b[2] - a[2]*b[1]
    v[1] = a[2]*b[0] - a[0]*b[2]
    v[2] = a[0]*b[1] - a[1]*b[0]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef inline void inverse(double **A, double **A_inv):
    cdef double det_inv = 1.0 / (
        A[0][0] * (A[1][1]*A[2][2]-A[1][2]*A[2][1]) -
        A[0][1] * (A[1][0]*A[2][2]-A[1][2]*A[2][0]) +
        A[0][2] * (A[1][0]*A[2][1]-A[1][1]*A[2][0])
        )

    A_inv[0][0] = det_inv * (A[1][1]*A[2][2] - A[1][2]*A[2][1])
    A_inv[0][1] = det_inv * (A[0][2]*A[2][1] - A[0][1]*A[2][2])
    A_inv[0][2] = det_inv * (A[0][1]*A[1][2] - A[0][2]*A[1][1])
    A_inv[1][0] = det_inv * (A[1][2]*A[2][0] - A[1][0]*A[2][2])
    A_inv[1][1] = det_inv * (A[0][0]*A[2][2] - A[0][2]*A[2][0])
    A_inv[1][2] = det_inv * (A[0][2]*A[1][0] - A[0][0]*A[1][2])
    A_inv[2][0] = det_inv * (A[1][0]*A[2][1] - A[1][1]*A[2][0])
    A_inv[2][1] = det_inv * (A[0][1]*A[2][0] - A[0][0]*A[2][1])
    A_inv[2][2] = det_inv * (A[0][0]*A[1][1] - A[0][1]*A[1][0])
