# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False
# cython: initializedcheck=False

import numpy as np
cimport numpy as cnp

from libc.math cimport sqrt, acos, atan2

cnp.import_array()

ctypedef cnp.double_t DTYPE_t

cdef inline double _clamp(double x, double lo, double hi) nogil:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def bond_series(cnp.ndarray[DTYPE_t, ndim=3] traj, int i, int j):
    """Return distances over frames between beads i and j."""
    cdef Py_ssize_t n_frames = traj.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] out = np.empty(n_frames, dtype=np.float64)
    cdef Py_ssize_t f
    cdef double dx, dy, dz

    for f in range(n_frames):
        dx = traj[f, i, 0] - traj[f, j, 0]
        dy = traj[f, i, 1] - traj[f, j, 1]
        dz = traj[f, i, 2] - traj[f, j, 2]
        out[f] = sqrt(dx * dx + dy * dy + dz * dz)

    return out


def angle_series(cnp.ndarray[DTYPE_t, ndim=3] traj, int i, int j, int k):
    """Return angles (deg) over frames for i-j-k."""
    cdef Py_ssize_t n_frames = traj.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] out = np.empty(n_frames, dtype=np.float64)
    cdef Py_ssize_t f

    cdef double v1x, v1y, v1z
    cdef double v2x, v2y, v2z
    cdef double dot, n1, n2, cosang
    cdef double inv_norm
    cdef double rad2deg = 180.0 / np.pi

    for f in range(n_frames):
        v1x = traj[f, i, 0] - traj[f, j, 0]
        v1y = traj[f, i, 1] - traj[f, j, 1]
        v1z = traj[f, i, 2] - traj[f, j, 2]

        v2x = traj[f, k, 0] - traj[f, j, 0]
        v2y = traj[f, k, 1] - traj[f, j, 1]
        v2z = traj[f, k, 2] - traj[f, j, 2]

        dot = v1x * v2x + v1y * v2y + v1z * v2z
        n1 = sqrt(v1x * v1x + v1y * v1y + v1z * v1z)
        n2 = sqrt(v2x * v2x + v2y * v2y + v2z * v2z)

        inv_norm = 1.0 / (n1 * n2)
        cosang = dot * inv_norm
        cosang = _clamp(cosang, -1.0, 1.0)
        out[f] = acos(cosang) * rad2deg

    return out


def dihedral_series(cnp.ndarray[DTYPE_t, ndim=3] traj, int i, int j, int k, int l):
    """Return dihedrals (deg) over frames for i-j-k-l."""
    cdef Py_ssize_t n_frames = traj.shape[0]
    cdef cnp.ndarray[DTYPE_t, ndim=1] out = np.empty(n_frames, dtype=np.float64)
    cdef Py_ssize_t f

    cdef double b1x, b1y, b1z
    cdef double b2x, b2y, b2z
    cdef double b3x, b3y, b3z

    cdef double n1x, n1y, n1z
    cdef double n2x, n2y, n2z

    cdef double b2n, b2nx, b2ny, b2nz
    cdef double m1x, m1y, m1z
    cdef double x, y

    cdef double rad2deg = 180.0 / np.pi

    for f in range(n_frames):
        # b1 = pi - pj
        b1x = traj[f, i, 0] - traj[f, j, 0]
        b1y = traj[f, i, 1] - traj[f, j, 1]
        b1z = traj[f, i, 2] - traj[f, j, 2]

        # b2 = pj - pk
        b2x = traj[f, j, 0] - traj[f, k, 0]
        b2y = traj[f, j, 1] - traj[f, k, 1]
        b2z = traj[f, j, 2] - traj[f, k, 2]

        # b3 = pk - pl
        b3x = traj[f, k, 0] - traj[f, l, 0]
        b3y = traj[f, k, 1] - traj[f, l, 1]
        b3z = traj[f, k, 2] - traj[f, l, 2]

        # n1 = b1 x b2
        n1x = b1y * b2z - b1z * b2y
        n1y = b1z * b2x - b1x * b2z
        n1z = b1x * b2y - b1y * b2x

        # n2 = b2 x b3
        n2x = b2y * b3z - b2z * b3y
        n2y = b2z * b3x - b2x * b3z
        n2z = b2x * b3y - b2y * b3x

        # b2 normalized
        b2n = sqrt(b2x * b2x + b2y * b2y + b2z * b2z)
        b2nx = b2x / b2n
        b2ny = b2y / b2n
        b2nz = b2z / b2n

        # x = n1 · n2
        x = n1x * n2x + n1y * n2y + n1z * n2z

        # m1 = n1 x b2_norm
        m1x = n1y * b2nz - n1z * b2ny
        m1y = n1z * b2nx - n1x * b2nz
        m1z = n1x * b2ny - n1y * b2nx

        # y = m1 · n2
        y = m1x * n2x + m1y * n2y + m1z * n2z

        out[f] = atan2(y, x) * rad2deg

    return out
