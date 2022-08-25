import time
from timeit import timeit

import astropy.units
from astropy import units as u
from astropy.constants import G
from scipy.linalg import eig
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, brentq
from numbalsoda import lsoda_sig, lsoda
import matplotlib.pyplot as plt
from numba import njit, cfunc
import numpy as np
import numba as nb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .utils import *

@njit(fastmath=True)
def radii_U_J(mu, X):
    """ Takes mu and state vector and returns
        the radii to the primary and secondary bodies (r1, r2),
        the pseudopotential (U), and the Jacobi Constant (J)."""
    x, y, z, dx, dy, dz = X
    r1 = ((x + mu) ** 2 + y ** 2 + z ** 2) ** 0.5
    r2 = ((x - 1 + mu) ** 2 + y ** 2 + z ** 2) ** 0.5
    U = (0.5 * (x ** 2 + y ** 2)) + ((1 - mu) / r1) + (mu / r2)
    J = U - (0.5 * (dx ** 2 + dy ** 2 + dz ** 2))
    return r1, r2, U, J


@njit(fastmath=True)
def U_Hessian(mu, X):
    """ Computes the Hessian of the pseudopotential.
        Note that this matrix is symmetric over the diagonal.
        Required for determining the linearized system matrix (A)"""

    x, y, z, dx, dy, dz = X
    # Compute the Hessian of U to arrive at A, the system matrix (for integrating STM):
    Uxx = mu * ((3 * (mu + x - 1) ** 2) / ((mu + x - 1) ** 2 + y ** 2 + z ** 2) ** (5 / 2) -
                1 / ((mu + x - 1) ** 2 + y ** 2 + z ** 2) ** (3 / 2)) + \
          (1 - mu) * ((3 * (mu + x) ** 2) / ((mu + x) ** 2 + y ** 2 + z ** 2) ** (5 / 2) -
                      1 / ((mu + x) ** 2 + y ** 2 + z ** 2) ** (3 / 2)) + 1
    Uxy = (3 * (1 - mu) * y * (mu + x)) / ((mu + x) ** 2 + y ** 2 + z ** 2) ** (5 / 2) + \
          (3 * mu * y * (mu + x - 1)) / ((mu + x - 1) ** 2 + y ** 2 + z ** 2) ** (5 / 2)
    Uyy = mu * ((3 * y ** 2) / ((mu + x - 1) ** 2 + y ** 2 + z ** 2) ** (5 / 2) -
                1 / ((mu + x - 1) ** 2 + y ** 2 + z ** 2) ** (3 / 2)) + \
          (1 - mu) * ((3 * y ** 2) / ((mu + x) ** 2 + y ** 2 + z ** 2) ** (5 / 2) -
                      1 / ((mu + x) ** 2 + y ** 2 + z ** 2) ** (3 / 2)) + 1
    Uxz = (3 * (1 - mu) * z * (mu + x)) / ((mu + x) ** 2 + y ** 2 + z ** 2) ** (5 / 2) + \
          (3 * mu * z * (mu + x - 1)) / ((mu + x - 1) ** 2 + y ** 2 + z ** 2) ** (5 / 2)
    Uyz = (3 * (1 - mu) * y * z) / ((mu + x) ** 2 + y ** 2 + z ** 2) ** (5 / 2) + \
          (3 * mu * y * z) / ((mu + x - 1) ** 2 + y ** 2 + z ** 2) ** (5 / 2)
    Uzz = mu * ((3 * z ** 2) / ((mu + x - 1) ** 2 + y ** 2 + z ** 2) ** (5 / 2) -
                1 / ((mu + x - 1) ** 2 + y ** 2 + z ** 2) ** (3 / 2)) + \
          (1 - mu) * ((3 * z ** 2) / ((mu + x) ** 2 + y ** 2 + z ** 2) ** (5 / 2) -
                      1 / ((mu + x) ** 2 + y ** 2 + z ** 2) ** (3 / 2))
    return Uxx, Uxy, Uxz, Uyy, Uyz, Uzz


def linearized_CR3BP(mu, X):
    """ Returns the linear system matrix (A) at a given
        point in the state space (X). """
    x, y, z, dx, dy, dz = X
    if isinstance(mu, np.ndarray):
        mu = mu[0]
    if isinstance(X, list):
        X = np.array(X, np.float64)
    Uxx, Uxy, Uxz, Uyy, Uyz, Uzz = U_Hessian(mu, X)
    if len(X.shape) == 1 or (1 in X.shape):
        A = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                      0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                      Uxx, Uxy, Uxz, 0.0, 2.0, 0.0,
                      Uxy, Uyy, Uyz, -2.0, 0.0, 0.0,
                      Uxz, Uyz, Uzz, 0.0, 0.0, 0.0],
                     np.float64).reshape(6, 6)
    else:
        n = len(x)
        A = np.zeros(6 * 6 * n).reshape(6, 6, n)
        for i in range(n):
            A[:, :, i] = np.array(
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                 Uxx[i], Uxy[i], Uxz[i], 0.0, 2.0, 0.0,
                 Uxy[i], Uyy[i], Uyz[i], -2.0, 0.0, 0.0,
                 Uxz[i], Uyz[i], Uzz[i], 0.0, 0.0, 0.0],
                np.float64).reshape(6, 6)
    return A


@njit(fastmath=True)
def CR3BP_EOM(mu, X, aug=0.0):
    """ CR3BP equations of motion.
    i.e., acceleration in the synodic barycentric frame as a function
    of current state. These differential equations are integrated
    numerically to arrive at a trajectory in the CR3BP.
    X : ~numpy.ndarray (6,1) - nondimensionalized state
                                [x,y,z,dx,dy,dz] """
    x, y, z, dx, dy, dz = X
    a_x = aug
    r1, r2, U, J = radii_U_J(mu, X)
    ddx = 2 * dy + x - (1 - mu) * (x + mu) / (r1 ** 3) - \
          mu * (x - 1 + mu) / (r2 ** 3) + a_x
    ddy = -2 * dx + y - (1 - mu) * y / (r1 ** 3) - \
          mu * y / (r2 ** 3)
    ddz = -(1 - mu) * z / (r1 ** 3) - \
          mu * z / (r2 ** 3)
    return [ddx, ddy, ddz]


def BR4BP_SOOP_EOM(mu, t, X, m_sun, a_sun,
                   i_sun, th_sun_0, ldn_sun):
    x, y, z, dx, dy, dz = X
    ddx_, ddy_, ddz_ = CR3BP_EOM(mu, X)
    ns = np.sqrt((1+m_sun)/(a_sun**3))
    ws = ns - 1
    th_sun = ws * t + th_sun_0
    xs = a_sun*(np.cos(th_sun-ldn_sun)*np.cos(ldn_sun) -
                np.sin(th_sun-ldn_sun)*np.sin(ldn_sun)*
                np.cos(i_sun))
    ys = a_sun*(np.cos(th_sun-ldn_sun)*np.sin(ldn_sun) +
                np.sin(th_sun-ldn_sun)*np.cos(ldn_sun)*
                np.cos(i_sun))
    zs = a_sun*np.sin(th_sun-ldn_sun)*np.sin(i_sun)
    rs = ((x-xs)**2 + (y-ys)**2 + (z-zs)**2)**0.5
    ddx = ddx_ - m_sun*s*(x-xs)/(rs**3) - \
          (m_sun/(a_sun**3))*xs
    ddy = ddy_ - m_sun*s*(y-ys)/(rs**3) - \
          (m_sun/(a_sun**3))*ys
    ddz = ddz_ - m_sun*s*(z-zs)/(rs**3) - \
          (m_sun/(a_sun**3))*zs

    return [ddx, ddy, ddz]


def lagrange_pts(mu, Lnum):
    def lagrange_eqns(r):
        x, y = r, 0
        r1 = np.sqrt((x + mu) ** 2 + y ** 2)
        r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2)
        return ((1 - mu) * (x + mu) / (r1 ** 3) +
                mu * (x - 1 + mu) / (r2 ** 3) - x)

    xL, yL = [0.0, 0.0]
    if Lnum in [1, 2, 3]:
        brackets = {1: (mu + 1e-11, 1 - mu - 1e-11),
                    2: (1, 1.5),
                    3: (-1.5, -1)}
        xL = brentq(lagrange_eqns, brackets[Lnum][0], brackets[Lnum][1])
        yL = 0
    elif Lnum in [4, 5]:
        xL = np.cos(np.pi / 3) - mu
        yL = np.sin(np.pi / 3) * (-2 * Lnum + 9)  # switches sign between 4/5
    return xL, yL


def augmented_L1(mu, ac):
    """ Computes the new equilibrium point corresponding to an object
        under constant radial acceleration. This can be used to account
        for solar radiation pressure at ESL1 or ESL2 for high-area-to-
        mass-ratio objects such as sailcraft."""
    X = np.array([0.9, 0, 0, 0, 0, 0], np.float64)
    def obj(x):
        X[0] = x
        x, y, z, dx, dy, dz = X
        ddx, ddy, ddz = CR3BP_EOM(mu, X)
        return abs(ddx + ac) + abs(y) + abs(dy) + abs(dx)
    xL_aug = fsolve(obj, X[0])[0]
    return xL_aug


@cfunc(lsoda_sig)
def cr3bp(t, X, dX, p):
    """ Unforced CR3BP equations of motion formulated as a cfunc for use by the
        numbalsoda IVP solver. Significantly faster
        p[0] : mu"""
    m = nb.carray(p, (1,))
    x, y, z, dx, dy, dz = nb.carray(X, (6,))
    r1 = ((x + m) ** 2 + y ** 2 + z ** 2) ** 0.5
    r2 = ((x - 1 + m) ** 2 + y ** 2 + z ** 2) ** 0.5
    dX[0] = dx
    dX[1] = dy
    dX[2] = dz
    dX[3] = 2 * dy + x - (1 - m) * (x + m) / (r1 ** 3) - m * (x - 1 + m) / (r2 ** 3)
    dX[4] = -2 * dx + y - (1 - m) * y / (r1 ** 3) - m * y / (r2 ** 3)
    dX[5] = -(1 - m) * z / (r1 ** 3) - m * z / (r2 ** 3)
cr3bp_ptr = cr3bp.address  # address to ODE function


@cfunc(lsoda_sig)
def cr3bp_aug(t, X, dX, p):
    """ Same as cr3bp() but augmented with a constant x-acceleration. Used in the shooting
        method within periodic() to determine a candidate periodic trajectory for a high-
        area-to-mass-ratio object, such as a sailcraft.
        p[0] : mu
        p[1] : a_x """
    m, a_x = nb.carray(p, (2,))
    x, y, z, dx, dy, dz = nb.carray(X, (6,))
    r1 = ((x + m) ** 2 + y ** 2 + z ** 2) ** 0.5
    r2 = ((x - 1 + m) ** 2 + y ** 2 + z ** 2) ** 0.5
    dX[0] = dx
    dX[1] = dy
    dX[2] = dz
    dX[3] = 2 * dy + x - (1 - m) * (x + m) / (r1 ** 3) - m * (x - 1 + m) / (r2 ** 3) + a_x
    dX[4] = -2 * dx + y - (1 - m) * y / (r1 ** 3) - m * y / (r2 ** 3)
    dX[5] = -(1 - m) * z / (r1 ** 3) - m * z / (r2 ** 3)
cr3bp_aug_ptr = cr3bp_aug.address  # address to ODE function


@cfunc(lsoda_sig)
def cr3bp_step(t, X, dX, p):
    """ Same as cr3bp() but the user passes step values as elements of the float64 array : p.
        p[0] : mu
        p[1] : baseline value of a_x
        p[2] : baseline value of a_y
        p[3] : baseline value of a_z
        p[4] : stepped value of a_x
        p[5] : stepped value of a_y
        p[6] : stepped value of a_z
        p[7] : time when a_x steps
        p[8] : time when a_y steps
        p[9] : time when a_z steps
        p[10] : time when a_x returns to baseline
        p[11] : time when a_y returns to baseline
        p[12] : time when a_z returns to baseline
        p[13] : the steepness, i.e. squareness of the steps (10.0 is a normal value)"""
    m = p[0]
    x, y, z, dx, dy, dz = nb.carray(X, (6,))
    a_x = smooth_step_single(t, p[1], p[4], p[7], p[10], p[13])
    a_y = smooth_step_single(t, p[2], p[5], p[8], p[11], p[13])
    a_z = smooth_step_single(t, p[3], p[6], p[9], p[12], p[13])

    r1 = ((x + m)**2 + y**2 + z**2)**0.5
    r2 = ((x - 1 + m)**2 + y**2 + z**2)**0.5
    dX[0] = dx
    dX[1] = dy
    dX[2] = dz
    dX[3] = 2*dy + x - (1 - m)*(x + m)/(r1**3) - m*(x - 1 + m)/(r2**3) + a_x
    dX[4] = -2*dx + y - (1 - m)*y/(r1**3) - m*y/(r2**3) + a_y
    dX[5] = -(1 - m)*z/(r1**3) - m*z/(r2**3) + a_z
cr3bp_step_ptr = cr3bp_step.address # address to ODE function


@njit(fastmath=True)
def sail_control(x, y, z, dx, dy, dz, x_tgt,
                 K_xy, K_z, db_xy, db_z):
    """ Control policy for rotating the clock angle of a refractive sailcraft about
        the sun line, which it remains oriented normal to at all times.
        Valid for small displacements from a collinear equilibrium point.

        x,y,z,dx,dy,dz : current state (full-state feedback)
        K_xy: Gain on in-plane position control.
        K_z: Gain on out-of-plane position control.
        db_xy: Deadband around the in-plane state feedback.
        db_z: Deadband around the out-of-plane state feedback.
        xL_aug: X coordinate of the equilibrium point, which is not L1 for HAMR objects.

        <> if x > xL_aug, and dx + dy is not favorable (not decreasing), apply acceleration
        in the -y direction (beta = pi).

        <> if x < xL_aug, and dx + dy is not favorable (not increasing), apply acceleration
        in the +y direction (beta = 0).

        <> When transiting between 0 and pi, turn in the direction that encourages staying in
        the equatorial plane, as determined by the value of z and the favoribility of dz."""

    x_err = x - x_tgt
    error = K_xy * deadband_cost(-x_err, 1e-3*db_xy,10,sym=1) - \
                    deadband_cost(dx + dy, db_xy,10,sym=1)
    switch = K_z * deadband_cost(z, 1e-5*db_z,10,sym=1) + \
                    deadband_cost(dz,db_z,10,sym=1)+ 1e-20
    beta = np.sign(switch) * (0.5*np.pi) * (1 - np.tanh(1e5*error))
    return beta, error


def sail_accel(X, ar, at, x_tgt, K_xy, K_z, db_xy, db_z):
    """" Takes the state, sail characteristics, and controller gains
         and returns the acceleration in cartesian coords at each step."""

    x,y,z,dx,dy,dz = X
    beta, error = sail_control(x,y,z,dx,dy,dz,x_tgt,
                               K_xy,K_z,db_xy,db_z)
    a_sph = np.array([ar, at * np.cos(beta), at * np.sin(beta)])
    a_x, a_y, a_z = sph2cart_fast(x, y, z, a_sph)
    return a_x[0], a_y[0], a_z[0]


def sail_control_diagnostic(X, ar, at, x_tgt, K_xy, K_z, db_xy, db_z):
    """ Non-jitted function for diagnostics on the control policy after simulation.
        Takes the entire state history as the input X, shape (6,n), the eq point, and gains."""
    x, y, z, dx, dy, dz = X
    beta = np.zeros(len(x))
    error = np.zeros_like(beta)
    for i in range(len(x)):
        beta[i], error[i] = sail_control(x[i],y[i],z[i],
                                         dx[i],dy[i],dz[i], x_tgt,
                                         K_xy, K_z, db_xy, db_z)
    a_sph = np.array([ar * np.ones(len(x)), at * np.cos(beta), at * np.sin(beta)])
    a_cart = sph2cart(x, y, z, a_sph)
    a_x, a_y, a_z = a_cart
    return a_x, a_y, a_z, error, beta


@cfunc(lsoda_sig)
def cr3bp_sail(t, X, dX, p):
    """ Differential equations describing the motion of a refractive sailcraft
        in the CR3BP that remains oriented normal to the sunline at all times,
        but has a clock angle beta about the sunline that determines the direction
        of its transverse acceleration.
        This is formulated as a @cfunc for compilation by numbalsoda, which significantly
        speeds up integration of the EOMs with control at every step.
        Current release of numbalsoda caps out at mxstep = 500, which is often too low
        for granular tolerances (below rtol = 1e-4 and atol = 1e-5, empirically).

        args (p): np.array([mu, ar, at, xL_aug, K_xy, K_z, db_xy, db_z])
        -------------------------------------------
        mu : CR3BP constant
        ar : radial acceleration of sail
        at : transverse acceleration of sail
        xL_aug : x coordinate of equilibrium point (for HAMR objects, this is shifted from L1)
        K_xy : Gain for in-plane position control
        K_z : Gain for out-of-plane position control
        db_xy : Deadband range for in-plane velocity feedback
        db_z : Deadband range for out-of-plane velocity feedback """
    m, ar, at, xL_aug, K_xy, K_z, db_xy, db_z = nb.carray(p, (8,))
    x, y, z, dx, dy, dz = nb.carray(X, (6,))
    beta, error = sail_control(x,y,z,dx,dy,dz,xL_aug,
                               K_xy, K_z, db_xy, db_z)
    a_sph = np.array([ar, at * np.cos(beta), at * np.sin(beta)])
    a_x, a_y, a_z = sph2cart_fast(x, y, z, a_sph)
    r1 = ((x + m) ** 2 + y ** 2 + z ** 2) ** 0.5
    r2 = ((x - 1 + m) ** 2 + y ** 2 + z ** 2) ** 0.5
    dX[0] = dx
    dX[1] = dy
    dX[2] = dz
    dX[3] = 2 * dy + x - (1 - m) * (x + m) / (r1 ** 3) - m * (x - 1 + m) / (r2 ** 3) + a_x
    dX[4] = -2 * dx + y - (1 - m) * y / (r1 ** 3) - m * y / (r2 ** 3) + a_y
    dX[5] = -(1 - m) * z / (r1 ** 3) - m * z / (r2 ** 3) + a_z
cr3bp_sail_ptr = cr3bp_sail.address  # address to ODE function


def propagation(mu, X0, thrust_prof=None, method='numbalsoda', rtol=1e-13,
                atol = 1e-16, t_max=2*np.pi, n=1000, aug=None):
    """
    mu ; ~float - CR3BP constant
    X0 : ~numpy.ndarray (6,) - Initial state (nondimensional)
    thrust_prof : ~str "[a_x(t,X), a_y(t,X), a_z(t,X)]" - evaluated during
                    integration.
    method : ~str - Integration method. Can use any method available for scipy's
             solve_ivp(), but the default is 'numbalsoda', which is a numba-
             accelerated version of solve_ivp()'s LSODA method, which is good
             for stiff ODEs. Note: numbalsoda currently limits the max # of
             integration steps to 500, which causes low-tolerance attempts to
             fail. A future release allows this to be changed. Also, a future
             release will include DOP853 as a numba-accelerated alternative to
             LSODA, which is a more precise integration method.
    rtol : ~float - Relative tolerance for integration.
    atol : ~float - Absolute tolerance for integration.
    t_max: ~float - End time of the simulation, in nondimensional time
    n : int - Number of timesteps desired
    aug : ~float - radial acceleration, if constant, i.e. for a HAMR object.
            Only used to calculate augmented equilibrium points and periodic
            trajectories under this assumption.
    """

    if aug is None:
        aug = 0.0

    def motion(t, X, thrust_prof):
        x, y, z, dx, dy, dz = X
        ddX = CR3BP_EOM(mu, X)
        if thrust_prof:
            a_x = eval(thrust_prof)[0]
            a_y = eval(thrust_prof)[1]
            a_z = eval(thrust_prof)[2]
        else:
            a_x = 0
            a_y = 0
            a_z = 0
        ddx = ddX[0] + a_x
        ddy = ddX[1] + a_y
        ddz = ddX[2] + a_z
        return [dx, dy, dz,
                ddx, ddy, ddz]

    interval = [0, t_max]
    initial_values = X0
    t_eval = np.linspace(0, t_max, n)
    if thrust_prof and method == 'numbalsoda':
        method = 'LSODA'
    if method != 'numbalsoda': # Use solve_ivp() - slower, more accurate
        sol = solve_ivp(motion, interval, initial_values, args=(thrust_prof,),
                        method=method, t_eval=t_eval, dense_output=True,
                        rtol=rtol, atol=atol)
        X = sol.y
    else: # Use numbalsoda - much faster
        p = np.array([mu, aug], np.float64)
        usol, success = lsoda(cr3bp_aug_ptr, X0, t_eval=t_eval, data=p,
                              rtol=rtol, atol=atol)
        X = np.array([usol[:, i] for i in range(6)])
    r1, r2, U, J = radii_U_J(mu, X)
    return t_eval, X, J


def periodic(mu, L_pt, eps0, zeta=0, d_eta0=None,
             aug=None, verbose=False):
    """ Use a shooting method to determine the initial conditions
        & period for a periodic orbit in the vicinity of a Lagrange
        Point (L_pt ~ int), given a small offset (delta ~ float, nondimensional)
        on the x-axis andout-of-plane offset (zeta ~ float, nondimensional). """

    xL, yL = lagrange_pts(mu, L_pt)
    if d_eta0 is None:
        # Compute linear approximation for initial velocity condition.
        # Here we are linearizing the EOM around the specified Lagrange Pt.
        #         ddX, J, A = CR3BP_EOM(mu, [xL, yL, 0, 0, 0, 0])
        A = linearized_CR3BP(mu, [xL, yL, 0, 0, 0, 0])
        A1 = np.delete(A, [2, 5], axis=1)
        A2 = np.delete(A1, [2, 5], axis=0)
        l, v = eig(A2)
        l3 = l[2]
        Uxx = A2[2, 0]
        a3 = (l3 ** 2 - Uxx) / (2 * l3)
        d_eta0 = np.real(a3 * eps0 * l3)

    # Initialize shooting method to find more precise value in the nonlinear dynamics.
    Rs_0 = [xL + eps0, 0, zeta]
    Vs_0 = [0, d_eta0, 0]
    u_0 = np.concatenate([Rs_0, Vs_0])
    if verbose:
        print('First-order (linear) guess: ', d_eta0)
        print('Numerically Integrated Solution: ')
        print('Y-Vel         Error\n-----         -----')

    if aug:
        p = np.array([mu, aug], np.float64)
        funcptr = cr3bp_aug_ptr
    else:
        funcptr = cr3bp_ptr

    #     @njit
    #     def motion(t, u_):
    #         dx,dy,dz = u_[3:]
    #         ddX, J, A = self.CR3BP_EOM(u_, aug=aug)
    #         return [dx, dy, dz, ddX[0], ddX[1], ddX[2]]

    def objective(v0):
        u_0[4] = v0
        t_max = 2 * np.pi
        t_eval = np.linspace(0, t_max, int(1e5))
        interval = [0, t_max]
        usol, success = lsoda(funcptr, u_0, t_eval=t_eval,
                              rtol=1e-13, atol=1e-16)
        x, y, dx = usol[:, 0], usol[:, 1], usol[:, 3]
        yswitch = np.sign(y[:-1]) == -np.sign(y[1:])
        i = np.where(yswitch == True)[0][0]  # index of the half-period point (first crossing)
        T = t_eval[2 * i]
        if verbose:
            print(v0, x[2 * i] - x[0])
        # return x[2*i] - x[0] # Enforce constraint: spacecraft returns to starting position
        return dx[i] ** 2 + dx[2 * i] ** 2 + abs(x[2 * i] - x[0])  # Enforce 0 x velocity at x crossings

    v0 = fsolve(objective, u_0[4])  # fsolve() outperforms root() and brentq() here.
    u_0[4] = v0[0]
    return u_0


class CR3BP:
    @u.quantity_input(m1='mass', m2='mass', d='length')
    def __init__(self, m1, m2, d):
        """
         m1 : float - mass of the primary body (kg)
         m2 : float - mass of the secondary body (kg)
         d  : float - (average) distance between the primaries (km)
        """
        self.G = G.to('km**3/(kg*s**2)')  # km will be the base unit for dim. length
        self.m1 = m1.to('kg')
        self.m2 = m2.to('kg')
        self.lc = d.to('km')  # Characteristic length, l*
        self.mc = (m1 + m2)  # Characteristic mass, m*
        self.tc = np.sqrt(self.lc ** 3 / (self.G * self.mc))  # Characteristic time, t*
        self.mu = (m2 / self.mc)  # Valuable dimensionless quantity in CR3BP
        self.X1 = self.lc * self.mu  # dimensional distance of primary from barycenter
        self.X2 = self.lc * (1 - self.mu)  # dimensional distance of secondary from barycenter
        self.xL = {'L%0.1i'%i: lagrange_pts(self.mu, i)[0] for i in range(1, 6)}
        self.yL = {'L%0.1i'%i: lagrange_pts(self.mu, i)[1] for i in range(1, 6)}

    def summary(self):
        """ Plots a sketch of the CR3BP with lagrange points.
            Also prints some of the class attributes. """

        x_m1 = -self.mu
        x_m2 = 1-self.mu
        # X1 = np.linspace(-x_m1, x_m1, 10000)
        X2 = np.linspace(-x_m2, x_m2, 10000)
        # Y1 = np.sqrt(x_m1**2 - X1**2)
        Y2 = np.sqrt(x_m2**2 - X2**2)

        fig, ax = plt.subplots()
        # ax.plot(X1, Y1, '--k')
        # ax.plot(X1, -Y1, '--k')
        ax.plot(X2, Y2, '--k')
        ax.plot(X2, -Y2, '--k')
        ax.plot([-1.2*x_m2, 1.2*x_m2], [0, 0], 'k')
        ax.plot([0,0], [-1.2 * x_m2, 1.2 * x_m2], 'k')
        ax.scatter(x_m2, 0, fc='k', s=50)
        ax.scatter(x_m1, 0, fc='k', s=150)
        ax.plot(self.xL['L1'], 0, 'kx')
        ax.text(self.xL['L1'], 0.05, 'L1')
        ax.plot(self.xL['L2'], 0, 'kx')
        ax.text(self.xL['L2'], 0.05, 'L2')
        ax.plot(self.xL['L3'], 0, 'kx')
        ax.text(self.xL['L3'], 0.05, 'L3')
        ax.plot(self.xL['L4'], self.yL['L4'], 'kx')
        ax.text(self.xL['L4'] + 0.05, self.yL['L4'] + 0.05, 'L4')
        ax.plot(self.xL['L5'], self.yL['L5'], 'kx')
        ax.text(self.xL['L5'] + 0.05, self.yL['L5'] - 0.05, 'L5')

        ax.scatter(0, 0, fc='w', ec='k')
        ax.set(xlim=[-1.2*x_m2, 1.2*x_m2], ylim=[-1.2*x_m2, 1.2*x_m2])
        ax.set_aspect('equal');

        print('CR3BP constant (μ): ', self.mu)
        print('  -  Note: μ is defined as m2 / (m1 + m2)')
        print('Primary body mass (m1): %0.6e kg' % self.m1.value)
        print('Secondary body mass (m2): %0.6e kg' % self.m2.value)
        print('Primary-Secondary distance: ', self.lc)
        print('L1 position (non-dimensional): %0.6f' % self.xL['L1'])
        print('L2 position (non-dimensional): %0.6f' % self.xL['L2'])

        return fig, ax

    def convert(self, X, to='nondim', state_vec=False):
        """
         x : float - quantity to convert between dimensional / nondimensional
         to : str - <'nondim', 'km', 'km/s', etc> - specify the direction of conversion.
            Important note: if a dimensional quantity is passed, this function
            assumes you want the nondimensional version, and vice versa. This
            is not for converting between dimensional units like m -> km. For that,
            use the appropriate astropy.units.Quantity method.
        """
        output = []
        n = 1
        if isinstance(X, list) or (isinstance(X,np.ndarray) and not isinstance(X,u.Quantity)):
            n = len(X)
            if isinstance(to, list):
                if len(to) != len(X):
                    if state_vec:
                        to = [to[0], to[0], to[0],
                              to[0] + '/' + to[1], to[0] + '/' + to[1], to[0] + '/' + to[1]]
                    else:
                        raise Exception('input list and destination unit list must have the same length.')
            else:
                to = [to for i in range(len(X))]
        else:
            if isinstance(to, list):
                raise Exception('input list and destination unit list must have the same length.')
            X = [X]
            to = [to]
        for i in range(len(X)):
            from_type = u.get_physical_type(X[i]).__str__()
            char_qtys = {'length': [self.lc, 'km'],
                         'speed/velocity': [(self.lc / self.tc), 'km/s'],
                         'acceleration': [(self.lc /
                                           (self.tc ** 2)), 'km/s2'],
                         'time': [self.tc, 's'],
                         'dimensionless': [1 * u.one, '']}
            s = str(1)
            if from_type == 'dimensionless':
                if to == 'nondim':
                    """ Already nondimensional """
                    output.append(X[i])
                else:
                    """ Return dimensionalized unit with specified physical type """
                    to_type = u.Unit(to[i]).physical_type.__str__()
                    output.append((X[i] * char_qtys[to_type][0]).to(to[i]))
            if u.get_physical_type(X[i]) in ['length', 'velocity', 'acceleration', 'time']:
                if to[i] == 'nondim':
                    """ Return nondimensionalized scaled version of the quantity """
                    X[i] = X[i].to(char_qtys[from_type][1])
                    output.append(X[i] * 1 / char_qtys[from_type][0])
                else:
                    """ already dimensional, execute conversion, if any"""
                    output.append(X[i].to(to[i]))
        if n == 1:
            return output[0]
        else:
            return output

    @u.quantity_input(Ri=u.km, Vi=u.km / u.s, R2=u.km)
    def inertial_to_synodic(self, Ri, Vi, R2):
        """ Ri : ~numpy.ndarray of length 3 - inertial position vector.
         Vi : ~numpy.ndarray of length 3 - inertial velocity vector.
         R2 : ~numpy.ndarray of length 3 - inertial position vector
                                           of the secondary body."""
        Ri = np.array(Ri).reshape(3, 1) * u.km
        Vi = np.array(Vi).reshape(3, 1) * u.km / u.s
        R2 = self.convert(R2, to='nondim')
        X1 = self.convert(self.X1, to='nondim')
        X2 = self.convert(self.X2, to='nondim')
        theta = np.arctan(R2[1] / (R2[0] - self.mu))
        dtheta = 1
        Qis = np.array([np.cos(theta), -np.sin(theta), 0,
                        np.sin(theta), np.cos(theta), 0,
                        0, 0, 1]).reshape(3, 3)
        dQis = dtheta * np.array([-np.sin(theta), -np.cos(theta), 0,
                                  np.cos(theta), -np.sin(theta), 0,
                                  0, 0, 0]).reshape(3, 3)
        Ri_nd = self.convert(Ri, to='nondim')
        Vi_nd = self.convert(Vi, to='nondim')
        Rs1 = np.linalg.inv(Qis) @ Ri_nd
        Vs = np.linalg.inv(Qis) @ (Vi_nd - dQis @ Rs1)
        Rs = np.array([Rs1[0] - self.mu, Rs1[1], Rs1[2]]) * u.one
        ts = theta
        return ts, Rs, Vs

    @u.quantity_input(Rs=u.one, Vs=u.one, ts=u.one)
    def synodic_to_inertial(self, Rs, Vs, ts):  # , ts):
        """ Ri : ~numpy.ndarray (3,1) - synodic position vector.
         Vi : ~numpy.ndarray (3,1) - synodic velocity vector.
         ts : float - nondimensionalized time."""
        Rs1 = np.array([Rs[0] + self.mu, Rs[1], Rs[2]]).reshape(3, 1)
        Vs = np.array(Vs).reshape(3, 1)
        theta = ts
        dtheta = 1
        Qis = np.array([np.cos(theta), -np.sin(theta), 0,
                        np.sin(theta), np.cos(theta), 0,
                        0, 0, 1]).reshape(3, 3)
        dQis = dtheta * np.array([-np.sin(theta), -np.cos(theta), 0,
                                  np.cos(theta), -np.sin(theta), 0,
                                  0, 0, 0]).reshape(3, 3)
        Ri_nd = Qis @ Rs1
        Vi_nd = Qis @ Vs + dQis @ Rs1
        Ri = self.convert(Ri_nd, to='km')
        Vi = self.convert(Vi_nd, to='km/s')
        return Ri, Vi

    def plot2D(self, X0, t_max=2*np.pi*u.one, n=1000, thrust_prof=None,
               style='-g', zoom=1, xlim=None, ylim=None, fig=None, ax=None, ax2=None):
        """
         Rs_0 : ~numpy.ndarray (3,1) - Initial position (nondimensional)
         Vs_0 : ~numpy.ndarray (3,1) - Initial velocity (nondimensional)
         t_max : float - Propagation time (nondimensional)
         n : int - Number of propagation timesteps
         style : str - specifying line plotting style/color (optional)
         zoom : float <[0,1]> - zoom factor for the trajectory plot.
         fig : matplotlib.pyplot.Figure - figure to plot on (optional)
         ax : matplotlib.pyplot.AxesSubplot - axes to plot on (optional)
         ax2 : matplotlib.pyplot.AxesSubplot - axes to plot on (optional)
         Plots the propagated motion in nondimensional units (left),
         Value(s) of the Jacobi constant (right),
         and returns the figure and axes (to plot on top of if desired).
         Variation in Jacobi constant should be on the order of the
         tolerance specified for solve_ivp() in the propagate() method.  """
        # Rs_0 = self.convert(X_0[:3]*u.km, to='nondim')
        # Vs_0 = self.convert(X_0[3:]*u.km_per_s, to='nondim')
        # t_max = self.convert(t_max*u.s, to='nondim')

        # if type(X0) != u.Quantity or type(t_max) != u.Quantity:
        #     print('Initial state is not an array of AstroPy Quantities. Proceeding\n'+\
        #           'under the assumption these are nondimensional values. If you\n'+\
        #           'meant to pass dimensional values (e.g. km, m/s, etc),\n'+\
        #           'you must do so as AstroPy Quantities.')
        # else:
        #     Rs_0 = self.convert(X0[:3].to('km'), to='nondim')
        #     Vs_0 = self.convert(X0[3:].to('km/s'), to='nondim')
        #     t_max = self.convert(t_max.to('s'), to='nondim')
        #     X0 = np.concatenate([Rs_0, Vs_0])

        prop_state = self.propagate(X0, t_max=t_max, n=n,
                                    thrust_prof=thrust_prof)
        T, X, J = prop_state

        if not (fig and ax):  # Create new fig and ax, none supplied
            fig = plt.figure()
            ax = fig.add_subplot(121)
            ax.scatter(-self.mu, 0, s=200, c='b', marker='o')  # Earth
            ax.scatter(1 - self.mu, 0, s=20, c='k', marker='o')  # Moon
            ax.set_aspect('equal')
            ax.grid('on')
            # Plot Lagrange Points
            Lx = [];
            Ly = []
            for i in range(1, 6):
                x, y = self.lagrange_pts(i)
                Lx.append(x);
                Ly.append(y)
            ax.scatter(Lx, Ly, c='k', marker='x')
            # Plot moon orbit
            x = np.linspace(-(1 - self.mu), (1 - self.mu), n)
            y = np.sqrt((1 - self.mu) ** 2 - x ** 2)
            ax.plot(x, y, '--k')
            ax.plot(x, -y, '--k')
            # Label plot
            t_max_dim = self.convert(t_max, to='day')
            ax.set_title('CR3BP propagated over %0.2f days'%t_max_dim.value)
            ax.set(xlabel='X (non-dimensional)',
                   ylabel='Y (non-dimensional)')
            # Plot Jacobi constant(s)
            ax2 = fig.add_subplot(122)
            ax2.set(title='Jacobi Constant(s)')
            ax2.grid('on')
            if xlim and ylim:
                ax.set(xlim=xlim, ylim=ylim)
            else:
                limx = (min(X[0]), max(X[0]))
                limy = (min(X[1]), max(X[1]))
                lim2 = (-1.2, 1.2)
                xlim_zoom = sigmoid(zoom) ** 0.1 * np.array(limx) + \
                            sigmoid(1 - zoom) * np.array(lim2)
                ylim_zoom = np.array(limy) + \
                            sigmoid(1 - zoom) * np.array(lim2)
                ax.set(xlim=xlim_zoom, ylim=ylim_zoom)
        #if ax:
        #    if xlim==None:
        #        xlim = ax.get_xlim()
        #    if ylim==None:
        #        ylim = ax.get_ylim()

        ax.plot(X0[0], X0[1], style + 'o')  # Initial position
        ax.plot(X[0], X[1], style)
        ax.set(xlim=xlim, ylim=ylim)
        # if xlim and ylim:
        #     ax.set(xlim=xlim, ylim=ylim)
        # else:
        #     limx = (min(X[0]), max(X[0]))
        #     limy = (min(X[1]), max(X[1]))
        #     lim2 = (-1.2, 1.2)
        #     xlim_zoom = sigmoid(zoom)**0.1 * np.array(limx) + \
        #                 sigmoid(1 - zoom) * np.array(lim2)
        #     ylim_zoom = np.array(limy) + \
        #                 sigmoid(1 - zoom) * np.array(lim2)
        #     ax.set(xlim=xlim_zoom, ylim=ylim_zoom)
        ax2.plot(J, style)
        ax2.set(ylim=[min(J) - 1e-2, max(J) + 1e-2])
        return fig, ax, ax2, prop_state

    def plotly2D(self, X0, t_max=2*np.pi*u.one, n=1000,
                 thrust_prof=None, tol=1e-13, method='DOP853',
                 bodies={'Primary':['Earth','blue'], 'Secondary':['Moon','gray']},
                 style='green', zoom=1, limits=None, fig=None):
        """
         Rs_0 : ~numpy.ndarray (3,1) - Initial position (nondimensional)
         Vs_0 : ~numpy.ndarray (3,1) - Initial velocity (nondimensional)
         t_max : float - Propagation time (nondimensional)
         n : int - Number of propagation timesteps
         style : str - specifying line plotting style/color (optional)
         zoom : float <[0,1]> - zoom factor for the trajectory plot.
         fig : matplotlib.pyplot.Figure - figure to plot on (optional)
         ax : matplotlib.pyplot.AxesSubplot - axes to plot on (optional)
         ax2 : matplotlib.pyplot.AxesSubplot - axes to plot on (optional)
         Plots the propagated motion in nondimensional units (left),
         Value(s) of the Jacobi constant (right),
         and returns the figure and axes (to plot on top of if desired).
         Variation in Jacobi constant should be on the order of the
         tolerance specified for solve_ivp() in the propagate() method.  """

        prop_state = self.propagate(X0, t_max=t_max, n=n, method=method,
                                    tol=tol, thrust_prof=thrust_prof)
        T, X_out, J = prop_state
        Ux = np.zeros_like(T)
        Uy = np.zeros_like(T)
        Uz = np.zeros_like(T)
        if thrust_prof:
            for i in range(n):
                X = X_out[:,i]
                Ux[i], Uy[i], Uz[i] = eval(thrust_prof)
        X=X_out

        if not fig:  # Create new fig and ax, none supplied
            fig = make_subplots(rows=2,cols=1,row_heights=[0.8,0.2])
            primary_marker = go.Scatter(x=[-self.mu.value], y=[0], name=bodies['Primary'][0],
                     marker=dict([('color',bodies['Primary'][1]),('size',20)]))
            secondary_marker = go.Scatter(x=[1-self.mu.value], y=[0], name=bodies['Secondary'][0],
                     marker=dict([('color',bodies['Secondary'][1]),('size',10)]))
            fig.add_trace(primary_marker, row=1, col=1)
            fig.add_trace(secondary_marker, row=1, col=1)
            fig.update_yaxes(scaleanchor='x', scaleratio=1)
            # Plot Lagrange Points
            Lx = [];
            Ly = []
            for i in range(1, 6):
                x, y = self.lagrange_pts(i)
                Lx.append(x);
                Ly.append(y)
            fig.add_trace(go.Scatter(x=Lx,y=Ly, mode='markers', marker_symbol='x-thin',
                                     marker_line_width=2, name='Lagrange Points',
                                     marker_line_color='black', marker_size=7))
            # Plot moon orbit
            xm = np.linspace(-(1 - self.mu.value), (1 - self.mu.value), n)
            ym = np.sqrt((1 - self.mu.value) ** 2 - xm ** 2)
            moon_orbit = go.scatter.Line(color='black', dash='dot')
            fig.add_trace(go.Scatter(x=xm, y=ym, mode='lines',
                                     line=moon_orbit, showlegend=False))
            fig.add_trace(go.Scatter(x=xm, y=-ym, mode='lines',
                                     line=moon_orbit, showlegend=False))
            # Label plot
            t_max_dim = self.convert(t_max, to='day')
            fig.update_layout(title='CR3BP propagated over %0.2f days'%t_max_dim.value)
            if limits is None:
                limx = (min(X[0]), max(X[0]))
                limy = (min(X[1]), max(X[1]))
                lim2 = (-1.2, 1.2)
                xlim = sigmoid(zoom) ** 0.1 * np.array(limx) + \
                       sigmoid(1 - zoom) * np.array(lim2)
                ylim = np.array(limy) + \
                       sigmoid(1 - zoom) * np.array(lim2)
                limits = np.concatenate([xlim, ylim])
            xlim, ylim = limits[:2], limits[2:]
            fig.update_xaxes(range=xlim, row=1, col=1)
            fig.update_yaxes(range=ylim, row=1, col=1)
            fig.update_yaxes(scaleanchor='x', scaleratio=1, row=1, col=1)

        # Plot Trajectory
        traj = go.scatter.Line(color=style)
        pos_ctrl = go.scatter.Marker(color='red')
        neg_ctrl = go.scatter.Marker(color='blue')
        x_ctrl_pos = [X[0][i] for i in np.where(Uy > 0)[0]]
        y_ctrl_pos = [X[1][i] for i in np.where(Uy > 0)[0]]
        x_ctrl_neg = [X[0][i] for i in np.where(Uy < 0)[0]]
        y_ctrl_neg = [X[1][i] for i in np.where(Uy < 0)[0]]
        fig.add_trace(go.Scatter(x=[X0[0]], y=[X0[1]], mode='markers',
                                 marker={'color':style},name='Initial State'))
        fig.add_trace(go.Scatter(x=X[0], y=X[1], mode='lines', line=traj,
                                 name='Trajectory'))
        fig.add_trace(go.Scatter(x=x_ctrl_pos,y=y_ctrl_pos,mode='markers',
                                 marker=pos_ctrl,name='Positive Transverse Control'))
        fig.add_trace(go.Scatter(x=x_ctrl_neg,y=y_ctrl_neg,mode='markers',
                                 marker=neg_ctrl,name='Negative Transverse Control'))
        # Plot Jacobian Constant
        J_line = go.Scatter(x=self.convert(T,to='day'), y=J, mode='lines',
                            line={'color':style, 'width':1},
                            name='Jacobian Constant')
        fig.add_trace(J_line, row=2, col=1)
        fig.update_xaxes(fixedrange=True, row=2, col=1)
        return fig, prop_state

    def plotly3D(self, X0, t_max=2*np.pi*u.one, n=1000, thrust_prof=None,
                 bodies={'Primary':['Earth','blue'], 'Secondary':['Moon','gray']},
                 style='green', zoom=1, limits=None, fig=None):
        """
         Rs_0 : ~numpy.ndarray (3,1) - Initial position (nondimensional)
         Vs_0 : ~numpy.ndarray (3,1) - Initial velocity (nondimensional)
         t_max : float - Propagation time (nondimensional)
         n : int - Number of propagation timesteps
         style : str - specifying line plotting style/color (optional)
         zoom : float <[0,1]> - zoom factor for the trajectory plot.
         fig : matplotlib.pyplot.Figure - figure to plot on (optional)
         ax : matplotlib.pyplot.AxesSubplot - axes to plot on (optional)
         ax2 : matplotlib.pyplot.AxesSubplot - axes to plot on (optional)
         Plots the propagated motion in nondimensional units (left),
         Value(s) of the Jacobi constant (right),
         and returns the figure and axes (to plot on top of if desired).
         Variation in Jacobi constant should be on the order of the
         tolerance specified for solve_ivp() in the propagate() method.  """

        prop_state = self.propagate(X0, t_max=t_max, n=n,
                                    thrust_prof=thrust_prof)
        T, X, J = prop_state

        if not fig:  # Create new fig and ax, none supplied
            fig = make_subplots(rows=2, cols=1, row_heights=[0.8, 0.2],
                                specs=[[{'type': 'scene'}],
                                       [{'type': 'xy'}]])
            primary_marker = go.Scatter3d(x=[-self.mu.value], y=[0], z=[0],
                                          name=bodies['Primary'][0],
                                          marker={'color':bodies['Primary'][1],'size':7})
            secondary_marker = go.Scatter3d(x=[1-self.mu.value], y=[0], z=[0],
                                            name=bodies['Secondary'][0],
                                            marker={'color':bodies['Secondary'][1],'size':2})
            fig.add_trace(primary_marker, row=1, col=1)
            fig.add_trace(secondary_marker, row=1, col=1)
            # Plot Lagrange Points
            Lx = []
            Ly = []
            for i in range(1, 6):
                x, y = self.lagrange_pts(i)
                Lx.append(x)
                Ly.append(y)
            fig.add_trace(go.Scatter3d(x=Lx,y=Ly,z=np.zeros(5), mode='markers',
                                       name='Lagrange Points',
                                       marker={'symbol':'x','color':'white','size':3}))
            # Plot moon orbit
            xm = np.linspace(-(1 - self.mu.value), (1 - self.mu.value), n)
            ym = np.sqrt((1 - self.mu.value) ** 2 - xm ** 2)
            moon_orbit = go.scatter3d.Line(color='black', dash='dot')
            fig.add_trace(go.Scatter3d(x=xm, y=ym, z=np.zeros(len(xm)), mode='lines',
                                     line=moon_orbit, showlegend=False))
            fig.add_trace(go.Scatter3d(x=xm, y=-ym, z=np.zeros(len(xm)), mode='lines',
                                     line=moon_orbit, showlegend=False))
            if limits is None:
                # Create cube-shaped limits from data
                xyz_spans = [max(u)-min(u) for u in X[0:3]]
                long = 0.5 * max(xyz_spans)
                xyz_mids = [np.average(u) for u in X[0:3]]
                limx, limy, limz = [(mid - long, mid + long) for mid in xyz_mids]
                limx2 = np.array([-1.2, 1.2]) + np.array([xyz_mids[0],xyz_mids[0]])
                limy2 = np.array([-1.2, 1.2]) + np.array([xyz_mids[1],xyz_mids[1]])
                xlim = sigmoid(zoom) * np.array(limx) + \
                       sigmoid(1 - zoom) * limx2
                ylim = np.array(limy) + \
                       sigmoid(1 - zoom) * limy2
                limits = np.concatenate([xlim, ylim])
            xlim, ylim = limits[:2], limits[2:]
            cam_settings = dict(up=dict(x=0, y=0, z=1),
                                center=dict(x=0, y=0, z=0),
                                eye=dict(x=0, y=0, z=0.8))
            # Label plot
            t_max_dim = self.convert(t_max, to='day')
            fig.update_layout(title='CR3BP propagated over %0.2f days'%t_max_dim.value,
                              scene=dict(camera = cam_settings,
                                         xaxis={'range': xlim},
                                         yaxis={'range': ylim},
                                         zaxis={'range': limz}),
                              template='plotly_dark')

        # Plot Trajectory
        traj = go.scatter3d.Line(color=style)
        fig.add_trace(go.Scatter3d(x=[X0[0]], y=[X0[1]], z=[X0[2]], mode='markers',
                                 marker={'color':style,'size':2},name='Initial State'))
        fig.add_trace(go.Scatter3d(x=X[0], y=X[1], z=X[2], mode='lines', line=traj,
                                 name='Trajectory'))
        # Plot Jacobian Constant
        J_line = go.Scatter(x=self.convert(T,to='day'), y=J, mode='lines',
                            line={'color':style, 'width':1},
                            name='Jacobian Constant')
        fig.add_trace(J_line, row=2, col=1)
        fig.update_xaxes(fixedrange=True, row=2, col=1)
        fig.update_yaxes(range=[min(J)-1e-2, max(J)+1e-2], row=2, col=1)
        return fig, prop_state

    def plot3D(self, Rs_0, Vs_0, t_max=2 * np.pi * u.one, n=1000, thrust_prof=None,
               style='-g', zoom=1, xlim=None, ylim=None, zlim=None, fig=None, ax=None):
        """ Rs_0 : ~numpy.ndarray (3,1) - Initial position (nondimensional)
         Vs_0 : ~numpy.ndarray (3,1) - Initial velocity (nondimensional)
         t_max : float - Propagation time (nondimensional)
         n : int - Number of propagation timesteps
         style : str - specifying line plotting style/color (optional)
         zoom : float <[0,1]> - zoom factor for the trajectory plot.
         fig : matplotlib.pyplot.Figure - figure to plot on (optional)
         ax : matplotlib.pyplot.AxesSubplot3D - axes to plot on (optional)
         Plots the 3D propagated motion in nondimensional units.
         Returns figure and axes for more plotting if desired."""
        Rs_0 = self.convert(Rs_0, to='nondim')
        Vs_0 = self.convert(Vs_0, to='nondim')
        t_max = self.convert(t_max, to='nondim')
        if not (fig and ax):  # Create new fig and ax, none supplied
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(-self.mu, 0, 0, s=200, c='b', marker='o')  # Earth
            ax.scatter(1 - self.mu, 0, 0, s=20, c='k', marker='o')  # Moon
            # Plot Lagrange Points
            Lx = [];
            Ly = []
            for i in range(1, 6):
                x, y = self.lagrange_pts(i)
                Lx.append(x);
                Ly.append(y)
            ax.scatter(Lx, Ly, 0, c='k', marker='x')
            # Plot moon orbit
            x = np.linspace(-(1 - self.mu), (1 - self.mu), n)
            y = np.sqrt((1 - self.mu) ** 2 - x ** 2)
            ax.plot(x, y, 0, '--k')
            ax.plot(x, -y, 0, '--k')
            # Label plot
            t_max_dim = self.convert(t_max, to='day')
            ax.set_title('CR3BP propagated over %0.2f days'%t_max_dim.value)
            limits = [-1.2, 1.2]
            ax.set(xlabel='X (non-dimensional)',
                   ylabel='Y (non-dimensional)',
                   zlabel='Z (non-dimensional)')
        ax.plot(Rs_0[0], Rs_0[1], Rs_0[2], style[-1] + 'o')  # Initial position
        prop_state = self.propagate(Rs_0, Vs_0, t_max=t_max, n=n,
                                    thrust_prof=thrust_prof)
        [T, Rx, Ry, Rz, Vx, Vy, Vz, J] = prop_state
        ax.plot3D(Rx, Ry, Rz, style)
        limxyz = min(np.concatenate([Rx, Ry, Rz])), max(np.concatenate([Rx, Ry, Rz]))
        lim2 = (-1.2, 1.2)
        limx = sigmoid(zoom) * np.array(limxyz) + \
               sigmoid((1 - zoom)) * np.array(lim2)
        limy = sigmoid(zoom) * np.array(limxyz) + \
               sigmoid((1 - zoom)) * np.array(lim2)
        limz = sigmoid(zoom) * np.array(limxyz) + \
               sigmoid((1 - zoom)) * np.array(lim2)
        ax.set(xlim3d=limx, ylim3d=limy, zlim3d=limz)
        return fig, ax, prop_state


class Spacecraft:
    def __init__(self, frame, X0=np.array([1,0,0,0,0,0]), ar=0.0, at=0.0,
                 thrust_prof=None):
        self.frame = frame
        self.set_initial_state(X0)
        self.controller_params = None
        self.set_thrust_profile(thrust_prof)
        self.controller = None
        self.ar = ar
        self.at = at
        self.xL_aug = augmented_L1(frame.mu, ar)

    def set_initial_state(self, X0_new):
        if isinstance(X0_new, u.Quantity):
            X0_new = self.frame.convert(X0_new, to='nondim')
        if isinstance(X0_new, list) or isinstance(X0_new, np.ndarray):
            self.X0 = np.array(X0_new, np.float64).reshape(6,)
        else:
            print('X0 must be a list or numpy array with 6 elements.')
        self.X = self.X0

    def set_thrust_profile(self, thrust, *args):
        """ thrust : ~str
            special keyword: 'sail_control' """
        self.thrust_prof = thrust
        if thrust == 'sail_control':
            self.controller = 'sail_control'
            if len(args)==0:
                print('For sail control, pass the gains and deadbands as follows:\n >',
                      'Spacecraft.set_thrust_profile("sail_control", x_tgt, K_xy, K_z, db_xy, db_z')
                self.controller_params = None
            else:
                self.controller_params = [p for p in args]

    def propagate(self, propagator, t_max=2*np.pi, n=10**6):
        self.T, self.X = propagator.integrate(t_max, n)
        return self.T, self.X


class Propagator:
    def __init__(self, frame, spacecraft, method, rtol=1e-3, atol=1e-6):
        self.frame = frame
        self.spacecraft = spacecraft
        self.method = method
        self.X0 = spacecraft.X0
        self.set_tolerance(rtol, atol)
        self.output = None

    def set_tolerance(self, rtol=1e-3, atol=1e-6):
        self.rtol = rtol
        self.atol = atol

    def update_initial_state(self, update):
        if isinstance(update, u.Quantity):
            update = self.frame.convert(update, to='nondim')
        if isinstance(update,Spacecraft):
            self.X0 = update.X0
            self.spacecraft = update
            return
        elif isinstance(update,list):
            if len(update) == 6:
                self.X0 = np.array(update, np.float64)
                self.spacecraft.X0 = self.X0
        elif isinstance(update, np.ndarray):
            if len(update) == 6:
                self.X0 = update
                self.spacecraft.X0 = self.X0

    def integrate(self, t_max, n=10**5):
        self.update_initial_state(self.spacecraft)
        if isinstance(t_max, u.Quantity):
            t_max = self.frame.convert(t_max, to='nondim')

        def motion(t, X, thrust_prof):
            x, y, z, dx, dy, dz = X
            ddX = CR3BP_EOM(self.frame.mu, X)
            if thrust_prof:
                a_x = eval(thrust_prof)[0]
                a_y = eval(thrust_prof)[1]
                a_z = eval(thrust_prof)[2]
            else:
                a_x = 0
                a_y = 0
                a_z = 0
            ddx = ddX[0] + a_x
            ddy = ddX[1] + a_y
            ddz = ddX[2] + a_z
            return [dx, dy, dz,
                    ddx, ddy, ddz]

        interval = [0, t_max]
        initial_values = self.X0
        t_eval = np.linspace(0, t_max, n)
        if self.method == 'numbalsoda':
            if self.spacecraft.thrust_prof == 'sail_control':
                x_tgt, K_xy, K_z, db_xy, db_z = self.spacecraft.controller_params
                mu_thr = np.array([self.frame.mu, self.spacecraft.ar, self.spacecraft.at,
                                   x_tgt,
                                   K_xy, K_z, db_xy, db_z], np.float64)
                cfunc_ptr = cr3bp_sail_ptr
            elif self.spacecraft.thrust_prof is None:
                mu_thr = np.array([self.frame.mu], np.float64)
                cfunc_ptr = cr3bp_ptr
            usol, success = lsoda(cfunc_ptr, self.X0, t_eval, data=mu_thr,
                                  rtol=self.rtol, atol=self.atol)
            T = t_eval
            X = np.array([usol[:, i] for i in range(6)])
        else:
            pass



        return T, X



