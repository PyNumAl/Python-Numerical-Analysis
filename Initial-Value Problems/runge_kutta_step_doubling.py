import numpy as np
from numpy import inf, asfarray, linspace, zeros, zeros_like, fabs, maximum, sqrt, mean, square, hstack, vstack, atleast_1d
from warnings import warn
from scipy.interpolate import CubicHermiteSpline as CHS, BPoly

def RMS(x):
    out = sqrt(
        mean(
            square(x)
            )
        )
    return out

class RKSD:
    """
    Solve a scalar/system of 1st-order ODEs using Runge-Kutta (RK) methods from order 1 to 8:
    
        dy / dt = fun(t, y)
    
    Fixed-step or adaptive integration is available. When adaptive integration is used, a scheme
    of RKq(p) is performed using step-doubling [1], where q = order + 1 and p = order.
    Richardson extrapolation is used to obtain a higher-order estimate as well as an estimate
    of the local truncation error. 
    
    A "free" quintic hermite spline interpolant is provided for dense output. This is "freely" obtained
    because the solution and derivative values at the midpoint of each step --- that were computed
    during step-doubling --- are not thrown away, but stored and used for the estimation of the second
    derivatives at all the mesh points. In my numerical tests, this interpolant sufficiently interpolates
    the Runge-Kutta methods from order 1 to 6. For order 7 and 8, the interpolation error is at most
    three orders of magnitude greater than the integration error.
    
    It can be shown that for a p-th order Runge-Kutta method, one can get by with
    dense output of order p−1 [2].
    
    A robust PI stepsize control as described in [3-4] is implemented for adaptive stepsize control.
    
    
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
    
    method : string, optional
        The Runge-Kutta method to use:
            
            'RK1' : Euler's method
            'RK2' : Heun's method
            'RK3' : Kutta's 3rd-order method [5]
            'RK4' (default) : The classical 4th-order RK method
            'RK5' : Butcher's 5th-order RK method [6-7]
            'RK6' : 6th-order RK method [8]
            'RK7' : 7th-order RK method [7]
            'RK8' : 8th-order RK method due to Cooper and Verner [7]
    
    adaptive : {True, False}, optional
        Whether to perform adaptive or fixed-step integration. Adaptive integration is done via step-doubling.
    
    h : float
        The stepsize to be used for fixed-step integration. For adaptive integration, it is taken as the guess for
        the first step satisfying atol and rtol. h must be greater than zero but not greater than the interval width.
    
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
        
        sol : scipy.interpolate.BPoly instance
            The quintic hermite spline interpolant
            
        t : ndarray, shape (n_points,)
            Time points.
            
        y : ndarray, shape (n_points, n)
            Solution values at t.
        
        hacc : integer
            Number of accepted steps.
        
        hrej : integer
            Number of rejected steps.
    
    
    References
    --------
    [1] Wikipedia, Adaptive step size. https://en.wikipedia.org/wiki/Adaptive_step_size
    [2] J. M. (https://scicomp.stackexchange.com/users/127/j-m), Intermediate values (interpolation) after Runge-Kutta calculation, URL (version: 2013-05-24): https://scicomp.stackexchange.com/q/7363
    [3] GUSTAFSSON, K , LUNDH, M , AND SODERLIND, G. A PI stepslze control for the numerical solution of ordinary differential equations, BIT 28, 2 (1988), 270-287.
    [4] GUSTAFSSON,K. 1991. Control theoretic techniques for stepsize selection in explicit RungeKutta methods. ACM Trans. Math. Softw. 174 (Dec.) 533-554.
    [5] Jacob Bishop. 7.1.7-ODEs: Third-Order Runge-Kutta. (Sept. 21, 2013). Accessed: July 3, 2022. [Online Video]. Available: https://www.youtube.com/watch?v=iS3hsHGY1Ok&t=25s
    [6] Jacob Bishop. 7.1.9-ODEs: Butcher's Fifth-Order Runge-Kutta. (Sept. 21, 2013). Accessed: July 3, 2022. [Online Video]. Available: https://www.youtube.com/watch?v=soEj7YHrKyE&t=6s
    [7] J. C. Butcher (26 July 2016). Numerical Methods for Ordinary Differential Equations, Third Edition, https://onlinelibrary.wiley.com/doi/book/10.1002/9781119121534
    [8] Luther, H.A., An Explicit Sixth-Order Runge-Kutta Formula. https://www.ams.org/journals/mcom/1968-22-102/S0025-5718-68-99876-1/S0025-5718-68-99876-1.pdf
    """
    def __init__(self, fun, t_span, y0, method='RK4', adaptive=True, h=None, hmax=inf, hmin=0., rtol=1e-3, atol=1e-6, maxiter=10**5, args=None):
        
        fun, t_span, y0, h, method, adaptive, hmax, hmin, rtol, atol, maxiter, args = self._check_input(fun, t_span, y0, h, method, adaptive, hmax, hmin, rtol, atol, maxiter, args)
        
        self.C, self.A, self.B, self.order = self.rk_tableau(method)
            
        if adaptive:
            t, y, dydt, hacc, hrej = self._RK_adaptive(fun, t_span, y0, h, adaptive, hmax, hmin, rtol, atol, maxiter, args)
            self.hacc = hacc
            self.hrej = hrej
            
            k = y.shape[1]
            n = t.size-1
            tn, tn1 = t[0:n:2], t[2:n+2:2]
            hn = np.full((k, tn.size), tn1-tn).T
            yn, yn12, yn1 = y[0:n:2], y[1:n+1:2], y[2:n+2:2]
            dyn, dyn12, dyn1 = dydt[0:n:2], dydt[1:n+1:2], dydt[2:n+2:2]
            d2y = np.zeros_like(dydt[0:n+2:2])
            d2y[0:-1] = (
                (
                 -46*yn + 32*yn12 + 14*yn1
                 ) / hn**2
                -
                (
                 12*dyn + 16*dyn12 + 2*dyn1
                 ) / hn
                )
            d2y[-1] = (
                (
                 14*yn[-1] + 32*yn12[-1] - 46*yn1[-1]
                 ) / hn[-1]**2
                +
                (
                 2*dyn[-1] + 16*dyn12[-1] + 12*dyn1[-1]
                 ) / hn[-1]
                )
            Y = np.zeros((t[0:n+2:2].size, 3, k))
            Y[:,0,:] = y[0:n+2:2]
            Y[:,1,:] = dydt[0:n+2:2]
            Y[:,2,:] = d2y
            sol = BPoly.from_derivatives(t[0:n+2:2], Y)
            self.sol = sol
            self.t = t[0:n+2:2]
            self.y = y[0:n+2:2]
            
        else:
            t, y, dydt = self._RK_fixed(fun, t_span, y0, h)
            self.hacc = t.size-1
            self.hrej = 0
            self.sol = CHS(t, y, dydt)
            self.t = t
            self.y = y
    
    def __repr__(self):
        attrs = ['sol', 't', 'y', 'hacc', 'hrej']
        m = max( map(len, attrs) ) + 1
        return '\n'.join([a.rjust(m) + ': ' + repr(getattr(self,a)) for a in attrs
                          ])
    
    def _RK_fixed(self, fun, t_span, y0, h):
        """
        Perform fixed-step integration.
        """
        C, A, B = self.C, self.A, self.B
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
            _, y_new, dy_new = self.rk_step(fun, t[i], y[i], dydt[i], h, C, A, B)
            y[i+1] = y_new
            dydt[i+1] = dy_new
        
        return t, y, dydt
    
    def _RK_adaptive(self, fun, t_span, y0, h, adaptive, hmax, hmin, rtol, atol, maxiter, args):
        """
        Perform adaptive integration.
        """
        C, A, B, order = self.C, self.A, self.B, self.order
        
        SAFETY=0.9
        MAX_FACTOR=10
        MIN_FACTOR=0.1
        exponent= -1/(order+1)
        KI = -.7/(order+1)
        KP = .4/(order+1)
        
        t0, tf = t_span
        t = t0
        y = y0
        dy = fun(t0,y0)
        
        if h is None:
            h = self._compute_first_step(fun, t0, y0, dy, order, rtol, atol)
        
        t_old = t0
        y_old = y0
        dy_old = dy
        
        step_rejected = False
        first_step = True
        oldr = 0.
        hacc = 0
        hrej = 0
        
        for i in range(maxiter):
            
            h = max(min(hmax, h), hmin)
            if t_old + h > tf:
                h = tf - t_old
                tm, t_new, ym, y_new, fm, dy_new, LTE = self.rk_double_step(fun, t_old, y_old, dy_old, h, C, A, B, order)
                t = hstack((t, tm, t_new))
                y = vstack((y, ym, y_new))
                dy = vstack((dy, fm, dy_new))
                hacc += 1
                break
            
            tm, t_new, ym, y_new, fm, dy_new, LTE = self.rk_double_step(fun, t_old, y_old, dy_old, h, C, A, B, order)
            sc = atol + rtol * maximum(fabs(y_old), fabs(y_new))
            LTE_norm = RMS(LTE/sc)
            
            if LTE_norm < 1 or h==hmin:
                
                if LTE_norm==0:
                    FAC = MAX_FACTOR
                else:
                    if first_step:
                        first_step = False
                        FAC = min(MAX_FACTOR, SAFETY * LTE_norm ** exponent)
                    else:
                        FAC = min(MAX_FACTOR, SAFETY * LTE_norm ** KI * oldr ** KP)
                
                if step_rejected:
                    step_rejected = False
                    FAC = min(1., FAC)
                
                h *=FAC
                oldr = LTE_norm
                
                t = hstack((t, tm, t_new))
                y = vstack((y, ym, y_new))
                dy = vstack((dy, fm, dy_new))
                t_old = t_new
                y_old = y_new
                dy_old = dy_new
                hacc +=1
            else:
                step_rejected = True
                FAC=max(MIN_FACTOR, SAFETY * LTE_norm ** exponent)
                h *= FAC
                hrej +=1
        else:
            warn(f'Maximum number of iterations has been reached without successfully completing the integration of the interval {t_span} within the specified absolute and relative tolerances, {atol} and {rtol}.')
        return t, y, dy, hacc, hrej
    
    @staticmethod
    def _check_input(fun, t_span, y0, h, method, adaptive, hmax, hmin, rtol, atol, maxiter, args):
        
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
        
        if type(hmin)!=float and hmin>=hmax or hmin<0:
            raise ValueError('hmin must be nonnegative and strictly less than hmax.')
        
        if hmax!=inf and ( type(hmax)!=float or hmax<=hmin or hmax<=0 ):
            raise ValueError(f'If hmax is not {inf}, it must be strictly greater than hmin and zero.')
        
        if type(adaptive)!=bool:
            raise Exception('adaptive argument must be a boolean.')
        
        methods = [
            'RK1', 'RK2', 'RK3', 'RK4', 'RK5', 'RK6', 'RK7', 'RK8'
            ]
        if type(method) is not str:
            raise TypeError('method argument must be a string.')
        elif method not in methods:
            raise Exception(f'method must be in {methods}. If you want to try other Runge-Kutta methods, supply the tableau via the C, A, B, and order arguments.')
        
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
        
        return fun, t_span, y0, h, method, adaptive, hmax, hmin, rtol, atol, maxiter, args
    
    @staticmethod
    def _compute_first_step(fun, t0, y0, f0, order, rtol, atol):
        """
        Compute/estimate the first step for adaptive integration when an initial stepsize h is not given.
        Taken from https://github.com/scipy/scipy/blob/4cf21e753cf937d1c6c2d2a0e372fbc1dbbeea81/scipy/integrate/_ivp/common.py#L64
        """
        sc = atol + rtol * np.abs(y0)
        h0 = RMS(.01 * sc / ( atol + rtol * np.abs(f0) ) )
        y1 = y0 + h0*f0
        f1 = fun(t0 + h0, y1)
        fdot0 = (f1-f0)/h0
        TE = np.hstack(
            (
                RMS(h0 ** (order+1) * 100 * fdot0 / sc)
                ,
                RMS(h0*f0)
                )
            )
        if np.all(TE<=1e-12):
            h1 = max(1e-6, 1e-3*h0)
        else:
            h1 = h0 * TE.max() ** (-1/(1+order))
        return min(1e2*h0,h1)
    
    @staticmethod
    def rk_step(fun, t, y, f, h, C, A, B):
        stages = A.shape[0]
        states = len(y)
        K = np.zeros((stages,states))
        K[0] = f
        if stages > 1:
            for i in range(1,stages):
                K[i] = fun(t + C[i]*h, y + h * A[i,:i] @ K[:i])
        y_new = y + h * B @ K
        t_new = t + h
        f_new = fun(t_new,y_new)
        return t_new, y_new, f_new

    def rk_double_step(self, fun, t, y, f, h, C, A, B, order):
        _, ycap, _ = self.rk_step(fun, t, y, f, h, C, A, B)
        tm, ym, fm = self.rk_step(fun, t, y, f, .5*h, C, A, B)
        t_new, yhalf, _ = self.rk_step(fun, tm, ym, fm, .5*h, C, A, B)
        err = (yhalf - ycap) / (2**order - 1)
        y_new = yhalf + err
        f_new = fun(t_new,y_new)
        return tm, t_new, ym, y_new, fm, f_new, err
    
    @staticmethod
    def rk_tableau(method):
        if method=='RK1':
            order = 1
            C = np.array([0.])
            A = np.array([[0.]])
            B = np.array([1.])
        elif method=='RK2':
            order = 2
            C = np.array([0., 1.])
            A = np.array([
                [0.,0.],
                [1., 0.]
                ])
            B = np.array([1/2, 1/2])
        elif method=='RK3':
            order = 3
            C = np.array([0, 1/2, 1.])
            A = np.array([
                [0., 0., 0.],
                [1/2, 0., 0.],
                [-1., 2., 0.]
                ])
            B = np.array([1/6, 2/3, 1/6])
        elif method=='RK4':
            order = 4
            C = np.array([0, 1/2, 1/2, 1.])
            A = np.zeros((4,4))
            A[1,:1] = [1/2]
            A[2,:2] = [0, 1/2]
            A[3,:3] = [0, 0, 1.]
            B = np.array([1/6, 1/3, 1/3, 1/6])
        elif method=='RK5':
            order = 5
            C = np.array([0, 1/4, 1/4, 1/2, 3/4, 1])
            A = np.zeros((6,6))
            A[1,:1] = [1/4]
            A[2,:2] = [1/8, 1/8]
            A[3,:3] = [0, 0, 1/2]
            A[4,:4] = [3/16, -3/8, 3/8, 9/16]
            A[5,:5] = [-3/7, 8/7, 6/7, -12/7, 8/7]
            B = np.array([7/90, 0., 16/45, 2/15, 16/45, 7/90])
        elif method=='RK6':
            order = 6
            r21 = 21 ** .5
            C = np.array([0, 1, 1/2, 2/3, (7-r21)/14, (7+r21)/14, 1])
            A = np.zeros((7,7))
            A[1,:1] = [1.]
            A[2,:2] = [3/8, 1/8]
            A[3,:3] = np.array([8, 2, 8]) / 27
            A[4,:4] = np.array([
                3*(3*r21 - 7),
                -8*(7 - r21),
                48*(7 - r21),
                -3*(21 - r21)
                ]) / 392
            A[5,:5] = np.array([
                -5*(231 + 51*r21),
                -40*(7 + r21),
                -320*r21,
                3*(21 + 121*r21),
                392*(6 + r21)
                ]) / 1960
            A[6,:6] = np.array([
                15*(22 + 7*r21),
                120,
                40*(7*r21 - 5),
                -63*(3*r21 - 2),
                -14*(49 + 9*r21),
                70*(7 - r21)
                ]) / 180
            B = np.array([9, 0, 64, 0, 49, 49, 9]) / 180
        elif method=='RK7':
            order = 7
            C = np.array([
                0, 1/6, 1/3, 1/2, 2/11, 2/3, 6/7, 0, 1
                ])
            A = np.zeros((9,9))
            A[1,:1] = [1/6]
            A[2,:2] = [0, 1/3]
            A[3,:3] = [1/8, 0, 3/8]
            A[4,:4] = [148/1331, 0, 150/1331, -56/1331]
            A[5,:5] = [-404/243, 0, -170/27, 4024/1701, 10648/1701]
            A[6,:6] = [2466/2401, 0, 1242/343, -19176/16807, -51909/16807, 1053/2401]
            A[7,:7] = [5/154, 0, 0, 96/539, -1815/20384, -405/2464, 49/1144]
            A[8,:8] = [-113/32, 0, -195/22, 32/7, 29403/3584, -729/512, 1029/1408, 21/16]
            B = np.array([
                0, 0, 0, 32/105, 1771561/6289920, 243/2560, 16807/74880, 77/1440, 11/270
                ])
        elif method=='RK8':
            order = 8
            r21 = 21 ** .5
            C = np.array([
                0, 1/2, 1/2, (7+r21)/14, (7+r21)/14, 1/2, (7-r21)/14, (7-r21)/14, 1/2, (7+r21)/14, 1
                ])
            A = np.zeros((11,11))
            A[1,:1] = [1/2]
            A[2,:2] = [1/4, 1/4]
            A[3,:3] = [1/7, -(7+3*r21)/98, (21+5*r21)/49]
            A[4,:4] = [(11+r21)/84, 0, (18+4*r21)/63, (21-r21)/252]
            A[5,:5] = [(5+r21)/48, 0, (9+r21)/36, (-231+14*r21)/360, (63-7*r21)/80]
            A[6,:6] = [(10-r21)/42, 0, (-432+92*r21)/315, (633-145*r21)/90, (-504+115*r21)/70,
                       (63-13*r21)/35]
            A[7,:7] = [1/14, 0, 0, 0, (14-3*r21)/126, (13-3*r21)/63, 1/9]
            A[8,:8] = [1/32, 0, 0, 0, (91-21*r21)/576, 11/72, -(385+75*r21)/1152, (63+13*r21)/128]
            A[9,:9] = [1/14, 0, 0, 0, 1/9, -(733+147*r21)/2205, (515+111*r21)/504, -(51+11*r21)/56,
                       (132+28*r21)/245]
            A[10,:10] = [0, 0, 0, 0, (-42+7*r21)/18, (-18+28*r21)/45, -(273+53*r21)/72, (301+53*r21)/72,
                         28*(1-r21)/45, (49-7*r21)/18]
            B = np.array([
                1/20, 0, 0, 0, 0, 0, 0, 49/180, 16/45, 49/180, 1/20
                ])
        return C, A, B, order
