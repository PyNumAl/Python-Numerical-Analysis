import numpy as np
from scipy.interpolate import CubicHermiteSpline as CHS, BPoly
from warnings import warn

class ABM:
    """
    Solve nonstiff initial-value problems using Adams-Bashforth-Moulton predictor-corrector methods.
    
    This class provides for numerically integrating a system of ordinary differential equations
    given an initial value:
        
        dy / dt = f(t, y)
        y(t0) = y0
        
    The Adams-Bashforth-Moulton (ABM) methods, with the exception of 1st-order ABM,
    require "bootstrapping" in order to use them. For a p-th order ABM method, a p-1-th order
    Runge-Kutta method is sufficient to calculate starting values [1], [2]. 1st-order ABM
    method doesn't require bootstrapping since it is just forward and backward euler methods
    as predictor-corrector pair which require no previous starting values.
    
    
    Parameters
    --------
    f : callable
        Function evaluating the 1st derivatives of y. Its calling signature must be ```f(t,y)```
    
    tspan : 2-tuple or 2-list of int or floats
        Interval of integration (t0, tf). The solver starts with t=t0 and integrates until it reaches t=tf.
    
    y0 : array_like, shape (n,)
        Initial state. If it is a scalar, it is promoted to a 1d array of size 1.
    
    h : int or float
        The fixed stepsize to use for integration.
    
    order : {1, 2, 3, 4, 5}, optional
        The order of the linear multistep predictor-corrector methods to use. Default is 4th-order.
    
    corrector : int, optional
        The number of corrector iterations to perform. Default is 1. The corrector iterations are exited when
        convergence has been reached.
    
    dense_output : boolean, optional
        Determines whether to output a continuous spline interpolant or not. If True and order < 5, then the continuous
        interpolant is a "free" cubic hermite spline interpolant. If True and order==5, the initial-value problem is re-solved
        using half the original stepsize h in order to obtain solution and derivative values
        at the midpoints of each subinterval. This allows for the construction of a quintic hermite spline interpolant
        so that the interpolant is of the same order as the multistep method. In addition, richardson extrapolation
        is performed on the solution and derivative values in order to not "waste" the more accurate approximations
        obtained with using twice as many subintervals. As a result, the accuracy of the approximations may be actually be
        6th-order, not just fifth, which is still well-interpolated by the 5th-degree interpolant.
        
    args : list or tuple, optional
        Additional arguments to pass to the user-defined function f. For example, if f has the signature f(t, y, a, b, c),
        then args must be a tuple/list of length 3.
    
    
    Returns
    --------
    Bunch object with the following fields defined:
    sol : `scipy.interpolate._cubic.CubicHermiteSpline`, `scipy.interpolate._interpolate.BPoly`, or None
        The continuous callable spline interpolant. None if dense_output is set to False.
    
    t : ndarray, shape (n_points,)
        Time points.
    
    y : ndarray, shape (n, n_points)
        Values of the solution at `t`.
    
    
    References
    --------
    [1] Lutz Lehmann (https://math.stackexchange.com/users/115115/lutz-lehmann), What is meant by Adams Bashforth being a "boot strap" method?, URL (version: 2016-10-03): https://math.stackexchange.com/q/1952179
    [2] Rackauckas, Chris. (Jun. 14, 2018). Crank-Nicolson Admas Bashforth (ABCN2). [Online]. Available at: https://github.com/SciML/OrdinaryDiffEq.jl/pull/384#discussion_r195180564
    """
    def __init__(self, f, tspan, y0, h, order=4, corrector=1, dense_output=True, args=None):
        
        f, tspan, y0 = self.validate_input(f, tspan, y0, h, order, corrector, dense_output, args)
        t, y, v = self.abm(f, tspan, y0, h, order, corrector)
        self.t = t
        self.y = y.T
        
        if dense_output:
            self.sol = self.spline_interpolant(f, tspan, y0, h, order, corrector, t, y, v)
        else:
            self.sol = None
        
    def __repr__(self):
        attrs = ['sol','t','y']
        m = max(map(len,attrs)) + 1
        return  '\n'.join([a.rjust(m) + ': ' + repr(getattr(self,a)) for a in attrs])
    
    def abm(self, f, tspan, y0, h, order, corrector):
        states = y0.size
        a, b = tspan
        n = int((b-a)/h)
        t, dt = np.linspace(a, b, n+1, retstep=True)
        if dt!=h:
            warn(f'The actual stepsize used, dt={dt}, is not equal to h={h}. This probably means that the interval of integration is not divisible by h.')
        else:
            pass
        y = np.zeros((n+1,states))
        y[0] = y0
        v = np.zeros_like(y)
        v[0] = f(a,y0)
        for i in range(n):
            if order > 1:
                if i < order-1:
                    y[i+1] = self.rk_step(f, t[i], y[i], v[i], h, order)
                else:
                    y[i+1] = self.abm_step(y, h, v, i, order, f, corrector, t)
            else:
                y[i+1] = self.abm_step(y, h, v, i, order, f, corrector, t)
            v[i+1] = f(t[i+1], y[i+1])
        return t, y, v
    
    def abm_step(self, y, h, v, i, order, f, corrector, t):
        yp = self.predict(y, h, v, i, order)
        yc = self.correct(y, h, v, i, order, yp, f, corrector, t)
        return yc
    
    @staticmethod
    def predict(y, h, v, i, order):
        if order==1:
            yp = y[i] + h * v[i]
        elif order==2:
            yp = y[i] + h/2 * (3*v[i] - v[i-1])
        elif order==3:
            yp = y[i] + h/12 * (23*v[i] - 16*v[i-1] + 5*v[i-2])
        elif order==4:
            yp = y[i] + h/24 * (55*v[i] - 59*v[i-1] + 37*v[i-2] - 9*v[i-3])
        else:
            yp = y[i] + h/720 * (1901*v[i] - 2774*v[i-1] + 2616*v[i-2] - 1274*v[i-3] + 251*v[i-4])
        return yp
    
    @staticmethod
    def correct(y, h, v, i, order, yp, f, corrector, t):
        for k in range(corrector):
            if order==1:
                yc = y[i] + h * f(t[i] + h, yp)
            elif order==2:
                yc = y[i] + h/2 * (f(t[i] + h, yp) + v[i])
            elif order==3:
                yc = y[i] + h/12 * (5*f(t[i] + h, yp) + 8*v[i] - v[i-1])
            elif order==4:
                yc = y[i] + h/24 * (9*f(t[i] + h, yp) + 19*v[i] - 5*v[i-1] + v[i-2])
            else:
                yc = y[i] + h/720 * (251*f(t[i] + h, yp) + 646*v[i] - 264*v[i-1] + 106*v[i-2] - 19*v[i-3])
            if np.allclose(yp, yc, 1e-12, 1e-12):
                break
            else:
                yp = yc
        return yc
    
    @staticmethod
    def rk_step(f, ti, yi, vi, h, order):
        k1 = vi
        if order==1:
            ynew = yi + h * k1
        elif order==2:
            k2 = f(ti + h, yi + h*k1)
            ynew = yi + h/2 * (k1 + k2)
        elif order==3:
            k2 = f(ti + .5*h, yi + .5*h*k1)
            k3 = f(ti + h, yi + h*(-k1 + 2*k2) )
            ynew = yi + h/6*(k1 + 4*k2 + k3)
        else:
            k2 = f(ti + .5*h, yi + .5*h*k1)
            k3 = f(ti + .5*h, yi + .5*h*k2)
            k4 = f(ti + 1.*h, yi + 1.*h*k3)
            ynew = yi + h/6*(k1 + 2*(k2 + k3) + k4)
        return ynew
    
    def spline_interpolant(self, f, tspan, y0, h, order, corrector, t, y, v):
        if order < 5:
            sol = CHS(t, y.T, v.T, 1)
        else:
            _, y2, v2 = self.abm(f, tspan, y0, .5*h, order, corrector)
            y[:] = y2[::2] + (y2[::2] - y) / (2**order-1)
            v[:] = v2[::2] + (v2[::2] - v) / (2**order-1)
            
            steps = t.size - 1
            h = np.full( (y.shape[1], y.shape[0]-1), t[1:] - t[:-1]).T
            k = y.shape[1]
            
            y_mid = np.zeros((steps, k))
            y_mid[:] = y2[1::2]
            
            v_mid = np.zeros_like(y_mid)
            v_mid[:] = v2[1::2]
            
            d2y = np.zeros_like(v)
            idx = slice(0, steps, 1)
            idx2 = slice(1, steps+1, 1)
            d2y[:steps] = (
                (-46*y[idx] + 32*y_mid[idx] + 14*y[idx2]) / h[idx]**2
                -
                (12*v[idx] + 16*v_mid[idx] + 2*v[idx2]) / h[idx]
                )
            d2y[-1] = (14*y[-2] + 32*y_mid[-1] - 46*y[-1]) / h[-1]**2 + (2*v[-2] + 16*v_mid[-1] + 12*v[-1]) / h[-1]
            
            Y = np.zeros((t.size, 3, k))
            Y[:,0,:] = y
            Y[:,1,:] = v
            Y[:,2,:] = d2y
            sol = lambda x: BPoly.from_derivatives(t, Y)(x).T
        return sol
    
    @staticmethod
    def validate_input(f, tspan, y0, h, order, corrector, dense_output, args):
        if args is not None:
            if isinstance(args,list) or isinstance(args,tuple):
                if len(args) >= 1:
                    f = lambda t, y, f=f: np.atleast_1d( f(t,y,*args) )
                else:
                    raise ValueError('args must be a list or tuple having length at least 1.')
            else:
                raise TypeError('args must be an instance of list or tuple.')
        else:
            f = lambda t, y, f=f: np.atleast_1d( f(t,y) )
        
        if type(dense_output)!=bool:
            raise TypeError('dense_output argument must be a boolean.')
        else:
            pass
        
        if type(corrector)!=int:
            raise TypeError('corrector argument must be an integer.')
        elif corrector < 1:
            raise ValueError('corrector argument must be a positive integer.')
        else:
            pass
        
        if order not in [1, 2, 3, 4, 5]:
            raise Exception('Invalid value for order. Order must be 1, 2, 3, 4, or 5.')
        else:
            pass
        
        y0 = np.atleast_1d(y0)
        if y0.ndim > 1:
            raise ValueError('y0 must be a vector (1d).')
        elif y0.dtype==complex:
            raise ValueError('Complex integration is not directly supported. Another option always available is to rewrite your problem for real and imaginary parts separately.')
        elif y0.dtype not in [int,float]:
            raise TypeError(f'y0 must be real-valued i.e. its data type either {int} or {float}.')
        else:
            pass
        
        tspan = np.asarray(tspan)
        if tspan.ndim!=1 or tspan.size!=2:
            raise Exception('tspan must be a 2-tuple or 2-list containing the integration endpoints.')
        elif tspan.dtype not in [int,float]:
            raise TypeError(f'Invalid dtype for tspan. It must be {int} or {float}.')
        elif tspan[0]>=tspan[1]:
            raise ValueError('The initial time value must be strictly less than the final time value.')
        else:
            pass
        
        if type(h) not in [int,float]:
            raise TypeError('h must be an int or float')
        elif h<=0:
            raise ValueError('The stepsize h must be a positive value.')
        elif h>tspan[1]-tspan[0]:
            raise ValueError('The stepsize cannot be larger than the interval width.')
        else:
            pass
        
        fout = f(tspan[0],y0)
        if fout.ndim>1:
            raise Exception('Output of f must be a vector (1d).')
        elif fout.size!=y0.size:
            raise Exception('Output of f and y0 must have the same length.')
        elif fout.dtype==complex:
            raise ValueError('Complex integration is not directly supported. Another option always available is to rewrite your problem for real and imaginary parts separately.')
        elif fout.dtype not in [int,float]:
            raise TypeError(f'f0 must be a real-valued vector function i.e. its data type either {int} or {float}.')
        else:
            pass
        
        return f, tspan, y0
