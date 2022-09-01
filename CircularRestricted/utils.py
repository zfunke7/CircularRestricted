""" Utilities for L1_Sunshade.py
"""

import numpy as np
import numba as nb
from numba import njit


@njit('void(float64[:,:],float64[:,:],float64[:,:])')
def matmul(matrix1, matrix2, rmatrix):
    """ Credit to Anilsathyan7: a numba-fied matrix multiplication routine
        https://gist.github.com/anilsathyan7/0f9e3682132471790738f258021e4524
    """
    for i in range(len(matrix1)):
        for j in range(len(matrix2[0])):
            for k in range(len(matrix2)):
                rmatrix[i][j] += matrix1[i][k] * matrix2[k][j]


def sph2cart(x, y, z, v_sph):
    """ Takes a current position in cartesian coordinates : x, y, and z,
        and a vector in spherical coordinates (i.e., where the basis vectors
        are range [r], azimuth [theta], and elevation [phi] respectively)
        and returns the same vector now expressed in cartesian coordinates."""
    # out = np.zeros(shape=(3, 1))
    r = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    th = np.arctan2(y, x)
    ph = np.arcsin(z / r)
    cth = np.cos(th)
    sth = np.sin(th)
    cph = np.cos(ph)
    sph = np.sin(ph)
    C = np.zeros(3*3*len(x)).reshape(3,3,len(x))
    out = np.zeros(3*len(x)).reshape(3,len(x))
    for i in range(len(x)):
        C[:,:,i] = np.array([cth[i] * cph[i], -sth[i], cth[i] * sph[i],
                             sth[i] * cph[i],  cth[i], sth[i] * sph[i],
                             sph[i],           0,      -cph[i]]        ).reshape(3, 3)
        v_cart = C[:,:,i] @ v_sph[:,i].reshape(3,1)
        out[:,i] = v_cart.reshape(3)
    return out


@njit(fastmath=True)
def sph2cart_fast(x, y, z, v_sph):
    """ Takes a current position in cartesian coordinates : x, y, and z,
        and a vector in spherical coordinates (i.e., where the basis vectors
        are range [r], azimuth [theta], and elevation [phi] respectively)
        and returns the same vector now expressed in cartesian coordinates."""
    out = np.zeros(shape=(3, 1))
    r = (x ** 2 + y ** 2 + z ** 2) ** 0.5
    th = np.arctan2(y, x)
    ph = np.arcsin(z / r)
    cth = np.cos(th)
    sth = np.sin(th)
    cph = np.cos(ph)
    sph = np.sin(ph)
    C = np.array([cth * cph, -sth, cth * sph,
                  sth * cph, cth, sth * sph,
                  sph, 0, -cph]).reshape(3, 3)
    matmul(C, v_sph.reshape(3, 1), out)
    return out


@njit(fastmath=True)
def sigmoid(x, s=15.0):
    """
    Returns 0 when x=0 and 1 when x=1, with logistic-like behavior in between.
    s : ~float can be a real number but is undefined at zero; this is the steepness angle.
    as s -> 0, sigmoid(x) approaches y=x. as s -> +inf or -inf, the shape of sigmoid(x)
    approximates the shape of heaviside(x-0.5, 0.5).
    """
    if s == 0:
        raise Exception('Steepness angle (s) can be any real number except 0.')
    b = 1+np.exp(-s)
    c = 1+np.exp(s)
    a = b/(1-b/c)
    d = -a/c
    return a/(1+np.exp(s*(1-2*x))) + d


def smooth_step(x, y0, y1, x0, xf, s=10):
    """ A function that takes a timeseries and returns a corresponding
        array of values that is one or more continuous 'steps' (using tanh)
        at heights, on-times, and off-times specified by the user.

        x  : the horizontal axis value or array
        y0 : the baseline vertical axis value
        y1 : the stepped value or array of values
        x0 : the on-time (or array of on-times) for step(s)
        xf : the off-time (or array of off-times) for step(s)
        s  : the steepness factor of the step """
    if isinstance(y0, (list, np.ndarray)):
        raise Exception('y0 must not be a list or array.')
    if isinstance(x0, (list, np.ndarray)):
        if not isinstance(y1, (list, np.ndarray)):
            y1 = y1 * np.ones(len(x0))
    else:
        yL = np.tanh(s * (x - x0)) / 2 + 1
        yR = np.tanh(s * (x - xf)) / 2 + 1
        return y1 * (yL - yR) + y0
    y = np.zeros_like(x)
    for i in range(len(x0)):
        yL = np.tanh(s * (x - x0[i])) / 2 + 1
        yR = np.tanh(s * (x - xf[i])) / 2 + 1
        y += y1[i] * (yL - yR)
    y += y0
    return y

@njit(fastmath=True)
def smooth_step_single(x, y0, y1, x0, xf, s=10.0):
    """ x  : input value
        y0 : the baseline vertical axis value
        y1 : the stepped value
        x0 : the on-time for step
        xf : the off-time for step
        s  : the steepness factor of the step """
    yL = np.tanh(s*(x-x0))/2+1
    yR = np.tanh(s*(x-xf))/2+1
    return y1*(yL-yR) + y0


@njit(fastmath=True)
def deadband_cost(x, x_db, s, sym=2):
    """ This is a two-sided continuous approximation of the ramp function with
        a deadband region where the function is zero in the middle. User supplies
        the width of the deadband region (x_db) and the slope of the ramp after
        the deadband region (s). User can optionally modify the symmetry by
        supplying an even value to 'sym' for bilateral symmetry (i.e. strictly
        positive outputs), or an odd value to 'sym' for pinwheel symmetry.
        It is the numerical integral of the 'bump' function b(x):
        b(x) = 0, x < 0
        b(x) = exp(-1/(ax)), x >= 0
        so the continuous ramp r(x) = s*x*b(x).
        To get the double-sided, deadband behavior we shift & add:
        y(x) = s * (x-x_db) * b(x-x_db) + w * s * (x+x_db) * b(x+x_db)
        Where w = 1 for pinwheel symmetry and w = -1 for bilateral symmetry.
        """
    y = s * (x - x_db) * \
        (np.zeros_like(x) + (np.sign(x - x_db) + 1) / 2 * np.exp(-1 / (1e20 * (x - x_db)))) + \
        (2*np.mod(sym,2)-1)*s * (x + x_db) * \
        (np.zeros_like(x) + (-np.sign(x + x_db) + 1) / 2 * np.exp(-1 / (1e20 * (-x + x_db))))
    return y


def read_jpl_ephem(filename):
    with open(filename) as f:
        lines = f.readlines()
    SOE_ind = lines.index('$$SOE\n')
    SOE_ind = lines.index('$$EOE\n')
    Eph_lines = lines[SOE_ind+1:EOE_ind]
    eph = {}
    for e_line in Eph_lines:
        e = e_line.split(',')
        t = float(e[0])
        X = [float(x) for x in e[2:8]]
        eph[t] = X
    return eph

