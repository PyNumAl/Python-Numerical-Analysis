from numpy import inf, asfarray, linspace, zeros, zeros_like, fabs, maximum, sqrt, mean, square, hstack, vstack, atleast_1d
from warnings import warn
from scipy.interpolate import CubicHermiteSpline as CHS, BPoly

class RK:
    """
    Solve a scalar/system of 1st-order ODEs using Runge-Kutta methods from order 1 to 5:
    
        dy / dt = fun(t, y, *args)
    
    Fixed-step or adaptive integration is available. When adaptive integration is used, a scheme
    of RKq(p) is performed using step-doubling (https://en.wikipedia.org/wiki/Adaptive_step_size),
    where q = order + 1 and p = order. Richardson extrapolation is used to obtain a higher-order estimate
    as well as a local truncation error estimate. Thus, the methods become:
        
        order = 1 : RK2(1)
        order = 2 : RK3(2)
        order = 3 : RK4(3)
        order = 4 : RK5(4)
        order = 5 : RK6(5)
    
    A hermite spline interpolant is provided for dense output. It can be shown that for a p-th order Runge-Kutta method,
    one can get by with dense output of order p−1 (https://scicomp.stackexchange.com/questions/7362/intermediate-values-interpolation-after-runge-kutta-calculation).
    Hence, for methods of order < 5, a cubic hermite spline (https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicHermiteSpline.html)
    is used. Otherwise, for methods of order >=5  e.g. RK5, RK5(4), and RK6(5), a quintic hermite spline using
    scipy.interpolate.BPoly.from_derivatives is created and used so that the interpolation error is of the same order
    as the method used.
    
    
    Parameters
    --------
    fun : callable
        ODE function returning the derivatives. Its calling signature must be ```fun(t,y)``` or
        ```fun(t,y,*args)``` if there are additional parameters. The output must either be a scalar
        or vector i.e. 1d array-like. If the output is a scalar, then it is converted into a 1d-array.
    
    t_span : 2-list or 2-tuple
        List or tuple containing the initial and final time values, the time span of the integration.
    
    y0 : 1d array_like
        The initial values. Must be a 1d input. Complex integration is not supported.
        An error will be raised when passed complex values.
    
    h : float
        The stepsize to be used for fixed-step integration. For adaptive integration, it is taken as the guess for
        the first step satisfying atol and rtol. h must be greater than zero but not greater than the interval width.
    order : {1, 2, 3, 4, 5}, optional
        Select which Runge-Kutta method to use.
        
        1 : Euler's method
        2 : Heun's method
        3 : 3rd-Order Runge-Kutta [1]
        4 : 4th-Order Runge-Kutta
        5 : Butcher's 5th-Order Runge-Kutta [2]
        
    adaptive : {True, False}, optional
        Whether to perform adaptive or fixed-step integration. Adaptive integration is done via step-doubling.
    
    hmax : {np.inf or float}, optional
        Maximum stepsize. Default is np.inf which means no upper bound on the stepsize. If a finite number,
        it must be strictly greater than hmin and greater than zero.
    
    hmin : float, optional
        Minimum stepsize. It must be strictly less than hmax and greater than zero to avoid getting stuck on singularities.
    
    rtol, atol : nonnegative float, optional
        Relative and absolute tolerances. Both rtol and atol cannot be zero or less than 1e-12 at the same time.
    
    maxiter : int, optional
        Maximum number of iterations for adaptive integration of the ODEs. A warning statement is printed when reached.
    
    args : {None, list or tuple}, optional
        List or tuple of additional parameters to be passed to the ODE function.
    
    
    Returns : Bunch object with the following fields defined:
        
        sol : The hermite spline interpolant.
        t : ndarray, shape (n_points,)
            Time points.
        y : ndarray, shape (n, n_points)
            Solution values at t.
    
    
    References
    --------
    [1]    Jacob Bishop. 7.1.7-ODEs: Third-Order Runge-Kutta. (Sept. 21, 2013). Accessed: July 3, 2022. [Online Video]. Available: https://www.youtube.com/watch?v=iS3hsHGY1Ok&t=25s
    [2]    Jacob Bishop. 7.1.9-ODEs: Butcher's Fifth-Order Runge-Kutta. (Sept. 21, 2013). Accessed: July 3, 2022. [Online Video]. Available: https://www.youtube.com/watch?v=soEj7YHrKyE&t=6s
    """
    def __init__(self, fun, t_span, y0, h=None, order=4, adaptive=True, hmax=inf, hmin=1e-4, rtol=1e-3, atol=1e-6, maxiter=10**5, args=None):
        
        fun, t_span, y0, h, order, adaptive, hmax, hmin, rtol, atol, maxiter, args = self._check_input(fun, t_span, y0, h, order, adaptive, hmax, hmin, rtol, atol, maxiter, args)
        
        if adaptive:
            t, y, dydt = self._RK_adaptive(fun, t_span, y0, h, order, adaptive, hmax, hmin, rtol, atol, maxiter, args)
        else:
            t, y, dydt = self._RK_fixed(fun, t_span, y0, h, order)
        
        self.sol = self._spline_interpolant(fun, t, y, dydt, order, adaptive)
        self.t = t
        self.y = y.T
    
    def __repr__(self):
        attrs = ['sol', 't', 'y']
        m = max( map(len, attrs) ) + 1
        return '\n'.join([a.rjust(m) + ': ' + repr(getattr(self,a)) for a in attrs
                          ])
    
    @staticmethod
    def _RK_step(fun, t_old, y_old, dy_old, h, order):
        """
        Perform a single runge-kutta step.
        """
        k1 = dy_old
        if order==1:
            # Euler's method
            y_new = y_old + h*k1
        elif order==2:
            # Heun's method
            k2 = fun(t_old + h, y_old + h*k1)
            y_new = y_old + .5*h*(k1 + k2)
        elif order==3:
            # RK3 [1]
            k2 = fun(t_old + .5*h, y_old + .5*h*k1)
            k3 = fun(t_old + h, y_old + h*(-k1 + 2*k2) )
            y_new = y_old + h/6*(k1 + 4*k2 + k3)
        elif order==4:
            # "Classical" 4th-Order Runge-Kutta method
            k2 = fun(t_old + .5*h, y_old + .5*h*k1)
            k3 = fun(t_old + .5*h, y_old + .5*h*k2)
            k4 = fun(t_old + 1.*h, y_old + h*k3)
            y_new = y_old + h/6*(k1 + 2*(k2 + k3) + k4)
        else:
            # Butcher's RK5 [2]
            k2 = fun(t_old + .25*h, y_old + .25*h*k1)
            k3 = fun(t_old + .25*h, y_old + h/8*(k1 + k2) )
            k4 = fun(t_old + .5*h, y_old + h*(-.5*k2 + k3) )
            k5 = fun(t_old + .75*h, y_old + h/16*(3*k1 + 9*k4) )
            k6 = fun(t_old + h, y_old + h/7*(-3*k1 + 2*k2 + 12*k3 - 12*k4 + 8*k5) )
            y_new = y_old + h/90*(7*k1 + 32*k3 + 12*k4 + 32*k5 + 7*k6)
        t_new = t_old + h
        dy_new = fun(t_new, y_new)
        return t_new, y_new, dy_new
    
    def _RK_fixed(self, fun, t_span, y0, h, order):
        """
        Perform fixed-step integration.
        """
        t0, tf = t_span
        n = int( (tf-t0)/h )
        t, dt = linspace(t0, tf, n + 1, retstep=True)
        
        if dt!=h:
            warn(f'Warning : the actual fixed stepsize used is {dt}, not {h}. This probably means that the interval width is not divisible by {h}.')
        
        k = y0.size
        y = zeros((n+1, k))
        y[0] = y0
        dydt = zeros_like(y)
        dydt[0] = fun(t0,y0)
        
        for i in range(n):
            _, y_new, dy_new = self._RK_step(fun, t[i], y[i], dydt[i], h, order)
            y[i+1] = y_new
            dydt[i+1] = dy_new
        
        return t, y, dydt
    
    def _RK_double_step(self, fun, t_old, y_old, dy_old, h, order):
        """
        Perform a more accurate step using half the stepsize, resulting in two/double steps.
        Used for adaptive steps.
        """
        t_half, y_half, dy_half = self._RK_step(fun, t_old, y_old, dy_old, h/2, order)
        t_new, y_new, dy_new = self._RK_step(fun, t_half, y_half, dy_half, h/2, order)
        return t_new, y_new, dy_new
    
    def _RK_adaptive_step(self, fun, t_old, y_old, dy_old, h, order):
        """
        Perform an adaptive runge-kutta step, with local truncation error (LTE) estimate.
        """
        # denote z as the lower accurate step
        # denote y as the more accurate double step
        t_new, z, _ = self._RK_step(fun, t_old, y_old, dy_old, h, order)
        _, y, _ = self._RK_double_step(fun, t_old, y_old, dy_old, h, order)
        # Local Truncation Error estimate
        LTE = (y - z) / (2**order-1)
        y_new = y + LTE
        dy_new = fun(t_new, y_new)
        return t_new, y_new, dy_new, LTE
    
    @staticmethod
    def _RMS(x):
        """
        Root-Mean-Square norm of a vector.
        """
        return sqrt(mean(square(x)))
    
    def _compute_first_step(self, fun, t0, y0, dy0, p, rtol, atol):
        """
        Compute/estimate the first step for adaptive integration when an initial stepsize h is not given.
        Taken from https://github.com/scipy/scipy/blob/4cf21e753cf937d1c6c2d2a0e372fbc1dbbeea81/scipy/integrate/_ivp/common.py#L64
        """
        scale = atol + fabs(y0) * rtol
        d0 = self._RMS(y0 / scale)
        d1 = self._RMS(dy0 / scale)
        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * d0 / d1

        y1 = y0 + h0 *dy0
        dy1 = fun(t0 + h0, y1)
        d2 = self._RMS((dy1 - dy0) / scale) / h0

        if d1 <= 1e-15 and d2 <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            h1 = (0.01 / max(d1, d2)) ** (1 / (p + 1))

        return min(100 * h0, h1)
    
    def _RK_adaptive(self, fun, t_span, y0, h, order, adaptive, hmax, hmin, rtol, atol, maxiter, args):
        """
        Perform adaptive integration.
        """
        SAFETY=0.9
        MAX_FACTOR=10
        MIN_FACTOR=0.2
        p, q = {1:(1,2), 2:(2,3), 3:(3,4), 4:(4,5), 5:(5,6)}[order]
        exponent= -1/q
        
        t0, tf = t_span
        t = t0
        y = y0
        dy = fun(t0,y0)
        
        if h is None:
            h = self._compute_first_step(fun, t0, y0, dy, p, rtol, atol)
        
        t_old = t0
        y_old = y0
        dy_old = dy
        for i in range(maxiter):
            if t_old + h > tf:
                h = tf - t_old
                t_new, y_new, dy_new, LTE = self._RK_adaptive_step(fun, t_old, y_old, dy_old, h, order)
                t = hstack((t, t_new))
                y = vstack((y, y_new))
                dy = vstack((dy, dy_new))
                break
            t_new, y_new, dy_new, LTE = self._RK_adaptive_step(fun, t_old, y_old, dy_old, h, order)
            scale = atol + rtol * maximum(fabs(y_old), fabs(y_new))
            LTE_norm=self._RMS(LTE/scale)
            if LTE_norm < 1 or h==hmin:
                if LTE_norm==0:
                    FAC = MAX_FACTOR
                else:
                    FAC = min(MAX_FACTOR, SAFETY * LTE_norm ** exponent)
                h *=FAC
                t = hstack((t, t_new))
                y = vstack((y, y_new))
                dy = vstack((dy, dy_new))
                t_old = t_new
                y_old = y_new
                dy_old = dy_new
            else:
                FAC=max(MIN_FACTOR, SAFETY * LTE_norm ** exponent)
                h *= FAC
            h = max(min(hmax, h), hmin)
        else:
            warn(f'Maximum number of iterations has been reached without successfully completing the integration of the interval {t_span} within the specified absolute and relative tolerances, {atol} and {rtol}.')
        return t, y, dy
    
    def _QuinticHermiteSpline(self, fun, t, y, dy, order):
        """
        Create and output a quintic hermite spline interpolant. This is used as the interpolant when the order of the method
        is greater than 4th order e.g. 5th order for fixed step integration and RK5(4) and RK6(5) for adaptive step integration.
        In order to create a quintic hermite spline, the 2nd derivatives at the solution values are needed. The solution and
        derivative values at the midpoint of each sub-interval are obtained by performing "midsteps".
        
        A quintic hermite spline can now be obtained via divided-differences (https://en.wikipedia.org/wiki/Hermite_interpolation).
        The expressions for the 2nd derivatives at either ends of a sub-interval are derived using SymPy as follows:
            
            (d2S / dt2)|t=0 = (-46*y[:-1] + 32*y_mid + 14*y[1:]) / h**2 - (12*dy[:-1] + 16*dy_mid + 2*dy[1:]) / h
            (d2S / dt2)|t=h = (14*y[:-1] + 32*y_mid - 46*y[1:]) / h**2 + (2*dy[:-1] + 16*dy_mid + 12*dy[1:]) / h
        
        """
        steps = t.size - 1
        h = t[1:] - t[:-1]
        k = y.shape[1]
        y_mid = zeros((steps, k))
        dy_mid = zeros_like(y_mid)
        d2y = zeros_like(dy)
        for i in range(steps):
            _, y_mid[i], dy_mid[i] = self._RK_step(fun, t[i], y[i], dy[i], .5*h[i], order)
            d2y[i] = (-46*y[i] + 32*y_mid[i] + 14*y[i+1]) / h[i]**2 - (12*dy[i] + 16*dy_mid[i] + 2*dy[i+1]) / h[i]
        d2y[-1] = (14*y[-2] + 32*y_mid[-1] - 46*y[-1]) / h[-1]**2 + (2*dy[-2] + 16*dy_mid[-1] + 12*dy[-1]) / h[-1]
        Y = zeros((t.size, 3, k))
        Y[:,0,:] = y
        Y[:,1,:] = dy
        Y[:,2,:] = d2y
        return lambda x: BPoly.from_derivatives(t, Y)(x).T
    
    def _spline_interpolant(self, fun, t, y, dy, order, adaptive):
        if adaptive:
            if order < 4:
                spl = CHS(t, y.T, dy.T, 1)
            else:
                spl = self._QuinticHermiteSpline(fun, t, y, dy, order)
        else:
            if order < 5:
                spl = CHS(t, y.T, dy.T, 1)
            else:
                spl = self._QuinticHermiteSpline(fun, t, y, dy, order)
        return spl
    
    @staticmethod
    def _check_input(fun, t_span, y0, h, order, adaptive, hmax, hmin, rtol, atol, maxiter, args):
        
        if args is not None:
            if isinstance(args, tuple) or isinstance(args, list) and len(args)>0:
                fun = lambda t, y, fun=fun: atleast_1d( fun(t, y, *args) )
            else:
                raise ValueError('args must be a tuple or list with at least length of 1.')
        else:
            fun = lambda t, y, fun=fun: atleast_1d( fun(t, y) )
        
        if type(maxiter)!=int or maxiter<1:
            raise ValueError('maxiter must be a positive integer.')
        
        if type(atol)!=float or atol<0:
            raise ValueError('atol must be a nonnegative float.')
        
        if type(rtol)!=float or rtol<0:
            raise ValueError('rtol must be a nonnegative float.')
        
        if rtol<1e-12 and atol<1e-12:
            warn(f'rtol and atol cannot be both zero or less than {1e-12}. Setting both to {1e-12}.')
            rtol, atol = [1e-12]*2
        
        if type(hmin)!=float or hmin>=hmax or hmin<=0:
            raise ValueError('hmin must be strictly less than hmax and strictly greater than zero.')
        
        if hmax!=inf and ( type(hmax)!=float or hmax<=hmin or hmax<=0 ):
            raise ValueError(f'If hmax is not {inf}, it must be strictly greater than hmin and zero.')
        
        if type(adaptive)!=bool:
            raise Exception('adaptive argument must be a boolean.')
        
        if order not in [1,2,3,4,5]:
            raise Exception('order must be 1, 2, 3, 4, or 5 only.')
        
        t_span = asfarray(t_span)
        if t_span.ndim!=1 or t_span.size!=2:
            raise ValueError('t_span must be a two-element 1d object containing the initial and final time values.')
        elif t_span[0]>=t_span[1]:
            raise ValueError('Invalid time span integration values. Final time value must be strictly greater than the initial time value.')
        else:
            pass
        
        if h is None:
            # Check if adaptive or fixed-step integration.
            if adaptive:
                pass
            else:
                raise Exception(f'h must not be {None} for fixed-step integration.')
        else:
            if type(h)!=float:
                raise Exception('h must be a float.')
            elif h<=0:
                raise ValueError('h must be a positive number.')
            elif h>(t_span[1] - t_span[0]):
                raise ValueError('h cannot be greater than the width of the integration interval.')
            else:
                pass
        
        y0 = asfarray(y0)
        if y0.ndim!=1:
            raise ValueError('y0 must be a vector (1d) of initial values.')
        
        fun_output = fun(t_span[0], y0)
        if fun_output.ndim!=1:
            raise Exception('The output of fun must be a scalar or vector (1d).')
        elif fun_output.dtype!=int and fun_output.dtype!=float:
            raise Exception('Output of fun can only be int or float.')
        else:
            pass
        
        return fun, t_span, y0, h, order, adaptive, hmax, hmin, rtol, atol, maxiter, args