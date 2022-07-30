import numpy as np
from numpy.linalg import norm
from warnings import warn
from scipy.interpolate import BPoly
EPS = np.finfo(float).eps

class RKNG34:
    """
    Solve general second-order scalar/vector initial-value problems of the form:
        
        d2y / dt2 = f(t, y, yp)
        
    This class implements Fine and Haute's 4th-order Runge-Kutta-Nystrom Generalized (RKNG)
    method as described in [1]. A "free" quintic spline interpolant is easily obtained and
    constructed using scipy.interpolate.BPoly.from_derivatives. The interpolant is 5th-order
    and 4th-order accurate for the position and velocity components, respectively, which are
    more than enough to provide a continuous extension of the solution.
    
    A robust PI stepsize control as described in [2-3] is implemented for adaptive stepsize control.
    
    
    Parameters
    --------
    fun : callable
        ODE function returning the 2nd derivatives. Its calling signature must be ```fun(t,y,yp)``` or
        ```fun(t,y,yp,*args)``` if there are additional parameters. The output must either be a scalar
        or vector i.e. 1d array-like. If the output is a scalar, then it is converted into a 1d-array.
        
    tspan : 2-list or 2-tuple
        List or tuple containing the initial and final time values, the time span of the integration.
    
    y0 : 1d array_like
        The position initial values. Scalar or 1d input. Scalar input is converted into a 1d-array.
        Complex integration is not supported. An error will be raised when passed complex values.
    
    yp0 : 1d array_like
        The position initial values. Scalar or 1d input. Scalar input is converted into a 1d-array.
        Complex integration is not supported. An error will be raised when passed complex values.
    
    h : integer or float, optional
        The initial stepsize to use.
    
    hmax : {np.inf or float}, optional
        Maximum stepsize. Default is np.inf which means no upper bound on the stepsize. If a finite number,
        it must be strictly greater than hmin and greater than zero.
    
    hmin : float, optional
        Minimum stepsize. If not given, then the stepsize has essentially no lower bound. The integration will be
        terminated and a ValueError raised when the stepsize h becomes very small i.e. smaller then 1000*EPS = 2.22e-13.
    
    rtol, atol : float or array_like, optional
        Relative and absolute tolerances. Default values are 1e-3 and 1e-6 respectively.
    
    maxiter : int, optional
        Maximum number of iterations for adaptive integration of the ODEs. A warning statement is printed when reached.
        Default is one hundred thousand (10**5) iterations.
    
    args : {None, list or tuple}, optional
        List or tuple of additional parameters to be passed to the ODE function.
    
    
    Returns : Bunch object with the following fields defined:
        
        t : ndarray, shape (n_points,)
            Time points.
            
        y : ndarray, shape (n_points, n)
            Solution values at t.
            
        sol : scipy.interpolate.BPoly instance
            Quintic hermite spline interpolant
            
        accepted_steps : integer
            The number of accepted steps
            
        rejected_steps : integer
            The number of rejected steps.
    
    
    References
    ---------
    [1] Fine, J.M. Low order practical Runge-Kutta-Nyström methods. Computing 38, 281–297 (1987). https://doi.org/10.1007/BF02278707
    [2] GUSTAFSSON, K , LUNDH, M , AND SODERLIND, G. A PI stepslze control for the numerical solution of ordinary differential equations, BIT 28, 2 (1988), 270-287.
    [3] GUSTAFSSON,K. 1991. Control theoretic techniques for stepsize selection in explicit RungeKutta methods. ACM Trans. Math. Softw. 174 (Dec.) 533-554.
    """
    def __init__(self, fun, tspan, y0, yp0, h=None, hmax=np.inf, hmin=0., rtol=1e-3, atol=1e-6, maxiter=10**5, args=None):
        y0, yp0, rtol, atol = self._validate_tol(y0, yp0, rtol, atol)
        
        tspan, h, hmax, hmin = self._validate_tspan(tspan, h, hmax, hmin)
        
        fun = self._validate_fun(fun, args)
        
        if type(maxiter)!=int or maxiter<1:
            raise ValueError('maxiter must be a positive integer.')
        
        t, y, yp, f, hacc, hrej = self._integrate(fun, tspan, y0, yp0, h, hmax, hmin, rtol, atol, maxiter)
        Y = np.zeros((t.size, 3, y.shape[1]))
        Y[:,0,:] = y
        Y[:,1,:] = yp
        Y[:,2,:] = f
        sol = BPoly.from_derivatives(t, Y)
        
        self.t = t
        self.y = y
        self.sol = sol
        self.accepted_steps = hacc
        self.rejected_steps = hrej
    
    def __repr__(self):
        attrs = ['t','y','sol','accepted_steps','rejected_steps']
        m = max(map(len,attrs)) + 1
        return '\n'.join([a.rjust(m) + ': ' + repr(getattr(self,a)) for a in attrs])
        
    def _integrate(self, fun, tspan, y0, yp0, h, hmax, hmin, rtol, atol, maxiter):
        MAX_FACTOR = 10.
        MIN_FACTOR = .1
        SAFETY = .9
        EXPONENT = -1/4
        KI = -.7/4
        KP = .4/4
        
        t0, tf = tspan
        f0 = fun(t0,y0,yp0)
        
        if h is None:
            h = self._compute_first_step(fun, t0, y0, yp0, f0, rtol, atol)
        
        t = t0
        y = y0.copy()
        yp = yp0.copy()
        f = f0.copy()
        
        t_old = t0
        y_old = y0.copy()
        yp_old = yp0.copy()
        f_old = f0.copy()
        
        step_rejected = False
        first_step = True
        oldr = 0.
        hacc = 0
        hrej = 0
        
        for i in range(maxiter):
            
            h = min(hmax, max(hmin, h) )
            if h < 1e3*EPS:
                raise ValueError(f'Integration is terminated due to the stepsize being too small, h={h}')
            
            if t_old + h > tf:
                h = tf - t_old
                t_new, y_new, yp_new, f_new, _ = self._rkng34_step(fun, t_old, y_old, yp_old, f_old, h, rtol, atol)
                t = np.hstack(( t, t_new ))
                y = np.vstack(( y, y_new ))
                yp = np.vstack(( yp, yp_new ))
                f = np.vstack(( f, f_new ))
                break
            
            t_new, y_new, yp_new, f_new, error_norm = self._rkng34_step(fun, t_old, y_old, yp_old, f_old, h, rtol, atol)
            
            if error_norm < 1:
                if error_norm == 0:
                    FAC = MAX_FACTOR
                else:
                    if first_step:
                        FAC = min(MAX_FACTOR, SAFETY * error_norm ** EXPONENT)
                        first_step = False
                    else:
                        FAC = min(MAX_FACTOR, SAFETY * error_norm ** KI * oldr ** KP)
                
                if step_rejected:
                    FAC = min(1,FAC)
                    step_rejected = False
                
                h *= FAC
                oldr = error_norm
                
                t = np.hstack(( t, t_new ))
                y = np.vstack(( y, y_new ))
                yp = np.vstack(( yp, yp_new ))
                f = np.vstack(( f, f_new ))
                
                t_old = t_new
                y_old = y_new
                yp_old = yp_new
                f_old = f_new
                
                hacc += 1
            else:
                h *= max(MIN_FACTOR, SAFETY * error_norm ** EXPONENT)
                step_rejected = True
                hrej += 1
        else:
            warn(f'Maximum number of iterations has been reached without successfully completing the integration of the interval {tspan} within the specified absolute and relative tolerances, {atol} and {rtol}.')
        
        return t, y, yp, f, hacc, hrej
    
    def _rkng34_step(self, fun, t, y, yp, f, h, rtol, atol):
        C, A, AP, B, E = self._rkng34_tableau()
        stages = 4
        states = len(y)
        F = np.zeros(( stages+1, states ))
        F[0] = f
        for i in range(stages):
            F[i+1] = fun(t + C[i]*h,
                         y + C[i]*h*yp + h**2 * A[i,:i+1] @ F[:i+1],
                         yp + h * AP[i,:i+1] @ F[:i+1]
                         )
        y_new = y + h*yp + h**2 * B[0] @ F
        yp_new = yp + h * B[1] @ F
        t_new = t + h
        f_new = fun(t_new, y_new, yp_new)
        err = np.array([
            h **2 * E[0] @ F
            ,
            h * E[1] @ F
            ])
        sc = np.array([
            atol + rtol * np.maximum( np.abs(y), np.abs(y_new) )
            ,
            atol + rtol * np.maximum( np.abs(yp), np.abs(yp_new) )
            ])
        error_norm = np.max(norm(err/sc, ord=np.inf, axis=1))
        return t_new, y_new, yp_new, f_new, error_norm
    
    @staticmethod
    def _rkng34_tableau():
        C = np.array([2/9, 1/3, 3/4, 1])
        A = np.array([
            [2/81, 0, 0, 0],
            [1/36, 1/36, 0, 0],
            [9/128, 0, 27/128, 0],
            [11/60, -3/20, 9/25, 8/75]
            ])
        AP = np.array([
            [2/9, 0, 0, 0],
            [1/12, 1/4, 0 ,0],
            [69/128, -243/128, 135/64, 0],
            [-17/12, 27/4, -27/5, 16/15]
            ])
        B = np.array([
            [19/180, 0, 63/200, 16/225, 1/120]
            ,
            [1/9, 0, 9/20, 16/45, 1/12]
            ])
        E = np.array([
            [25/1116, 0, -63/1240, 64/1395, -13/744],
            [2/125, 0, -27/625, 32/625, -3/125]
            ])
        return C, A, AP, B, E
    
    @staticmethod
    def _compute_first_step(fun, t0, y0, yp0, f0, rtol, atol):
        scale = np.array([
            atol + rtol * np.abs(y0),
            atol + rtol * np.abs(yp0),
            atol + rtol * np.abs(f0)
            ])
        b = norm(scale[1]/scale[2], np.inf)
        c = norm(scale[0]/scale[2], np.inf)
        h0 = np.array([
            .01*b,
            .5 * (-2*b + (b**2 + 8*c)**.5),
            .5 * (-2*b - (b**2 + 8*c)**.5)
            ])
        h0 = max(h0[h0>0].min(), 1e-6)
        
        y1 = y0 + h0 * yp0 + .5 * h0 ** 2 * f0
        yp1 = yp0 + h0 * f0
        f1 = fun(t0 + h0, y1, yp1)
        y3p = (f1 - f0) / h0
        
        norm1 = norm(y3p/scale[0], np.inf)
        norm2 = norm(y3p/scale[1], np.inf)
        
        if norm1<=1e-15 and norm2<1e-15:
            h1 = max(1e-6, 1e-3*h0)
        else:
            h1 = (.01 / max(norm1, norm2)) ** (1/4)
        return min(100 * h0, h1)
    
    @staticmethod
    def _validate_fun(fun, args):
        if args is not None:
            if isinstance(args, tuple) or isinstance(args, list) and len(args)>0:
                fun = lambda t, y, yp, fun=fun: np.atleast_1d( fun(t, y, yp, *args) )
            else:
                raise ValueError('args must be a tuple or list with at least length of 1.')
        else:
            fun = lambda t, y, yp, fun=fun: np.atleast_1d( fun(t, y, yp) )
        return fun
    
    @staticmethod
    def _validate_tspan(tspan, h, hmax, hmin):
        tspan = np.asarray(tspan)
        if tspan.ndim!=1 or tspan.size!=2:
            raise Exception('tspan must be a 2-tuple or 2-list containing the endpoints of the integration interval.')
        elif tspan.dtype not in [int,float]:
            raise TypeError('dtype of tspan must be int or float.')
        elif tspan[0]>=tspan[1]:
            raise ValueError('t0 must be strictly less than tf.')
        else:
            pass
        
        if h is not None:
            if type(h) not in [int,float]:
                raise TypeError('h must be an int or float')
            elif h<=0:
                raise ValueError('The stepsize h must be a positive value.')
            elif h>tspan[1]-tspan[0]:
                raise ValueError('The stepsize cannot be larger than the interval width.')
            else:
                pass
        
        if hmin!=None:
            if type(hmin)!=float and hmin>=hmax or hmin<0:
                raise ValueError(f'If hmin is not {None}, it must be strictly less than hmax and greater than or equal to zero.')
            else:
                pass
        else:
            pass
        
        if hmax!=np.inf and ( type(hmax)!=float or hmax<=hmin or hmax<=0 ):
            raise ValueError(f'If hmax is not {np.inf}, it must be strictly greater than hmin and zero.')
        
        return tspan, h, hmax, hmin
    
    @staticmethod
    def _validate_tol(y0, yp0, rtol, atol):
        y0 = np.atleast_1d(y0)
        if y0.ndim > 1:
            raise ValueError('y0 must be a vector (1d) of initial values.')
        elif y0.dtype not in [int,float]:
            raise TypeError('dtype of y0 must be int or float.')
        else:
            pass
        
        yp0 = np.atleast_1d(yp0)
        if yp0.ndim > 1:
            raise ValueError('yp0 must be a vector (1d) of initial values.')
        elif yp0.dtype not in [int,float]:
            raise TypeError('dtype of yp0 must be int or float.')
        else:
            pass
        
        if y0.size!=yp0.size:
            raise ValueError('Vectors y0 and yp0 must have compatible shapes.')
        
        rtol = np.asarray(rtol)
        if rtol.ndim > 1:
            raise ValueError('rtol must be a scalar or vector (1d) of compatible shape.')
        elif rtol.ndim==1 and rtol.size!=y0.size:
            raise ValueError('If rtol is a vector, it must have the same size as y0.')
        elif rtol.dtype not in [int, float]:
            raise TypeError('dtype of rtol must be int or float.')
        elif np.any(rtol<0):
            raise ValueError('rtol values must be nonnegative.')
        elif np.any(rtol<1e3*EPS):
            warn(f'rtol value/s too small, setting to {1e3*EPS}')
            rtol = np.where(rtol<1e3*EPS, 1e3*EPS, rtol)
        else:
            pass
        atol = np.asarray(atol)
        if atol.ndim > 1:
            raise ValueError('atol must be a scalar or vector (1d) of compatible shapes.')
        elif atol.ndim==1 and atol.size!=y0.size:
            raise ValueError('If atol is a vector, it must have the same size as y0.')
        elif atol.dtype not in [int, float]:
            raise TypeError('dtype of atol must be int or float.')
        elif np.any(atol<0):
            raise ValueError('atol values must be nonnegative.')
        else:
            pass
        return y0, yp0, rtol, atol