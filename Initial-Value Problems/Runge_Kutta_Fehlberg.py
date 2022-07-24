import numpy as np
from warnings import warn
from scipy.interpolate import BPoly

EPS = np.finfo(float).eps
def RMS(x):
    return np.sqrt(np.mean(np.square(x)))

class RKF:
    """
    Solve nonstiff initial-value problems using Runge-Kutta-Fehlberg methods.
    
    This class provides for numerically integrating a system of ordinary differential equations
    given an initial value:
        
        dy / dt = f(t, y)
        y(t0) = y0
    
    This class implements Fehlberg's RK45 [1] and RK78 [2] methods in local extrapolation mode i.e.
    the higher order formula is used to advance the solution even though they were originally intended
    to be used in RKF4(5) and RKF7(8) modes. Practical results indicate that this is preferable [3-5].
    Hence, the methods are RKF5(4) and RKF8(7) respectively.
    
    A hermite spline interpolant of fifth and seventh degree are used for dense output for the RK45
    and RK78 methods, respectively. The fifth-degree interpolant is constructed by obtaining the solution
    and derivative values at the midpoint of each subinterval while the seventh-degree interpolant
    obtains the solution and derivative values at the one-third and two-thirds step of each subinterval.
    Then the symbolic expressions for the second (and third derivatives for the seventh-degree spline)
    are obtained in terms of the calculated values:
        
        y_n : solution value at the left endpoint
        y_n+1/3 : solution value at 1/3 of the subinterval
        y_n+1/2 : solution value at the midpoint
        y_n+2/3 : solution value at 2/3 of the subinterval
        y_n+1 : solution value at the right endpoint
        
        f_n : derivative value at the left endpoint
        f_n+1/3 : derivative value at 1/3 of the subinterval
        f_n+1/2 : derivative value at the midpoint
        f_n+2/3 : derivative value at 2/3 of the subinterval
        f_n+1 : derivative value at the right endpoint
        
    Symbolic Python (SymPy) is used for this purpose. Finally, SciPy's BPoly.from_derivatives
    is used t0 create the higher-order splines after calculating the required higher derivatives
    at the nodes.
    
    A more robust PI stepsize control as described in [6-7] is implemented for adaptive stepsize control.
    
    
    IMPORTANT NOTE (1:16 PM, 2022-07-19) : 
        
        Fehlberg lists the error coefficients of his RK45 pair in [1] as:
            
            [-1/360, 0, 128/4275, 2187/75240, -1/50, -2/55]
        
        The fourth coefficient's numerator, 2187, is wrong. Subtracting the coefficients of the lower and higher order
        formulas yields 2197 / 75240 for the 4th error coefficient. Copying the error coefficients
        verbatim from the [1] caused the rkf45 method to be very inefficient for relative 
        tolerances smaller than .001 (or 1e-3). Thankfully, I already had a working version of RKF45
        that didn't utilize the error coefficients, but involved actually subtracting the lower and higher order
        estimates. This prior working version was crucial in allowing me to debug and find out
        what was causing the inefficiency of the RK45 pair when the RK78 pair was working quite well in my numerical tests.
    
    
    Parameters
    --------
    fun : callable
        ODE function returning the derivatives. Its calling signature must be ```fun(t,y)``` or
        ```fun(t,y,*args)``` if there are additional parameters. The output must either be a scalar
        or vector i.e. 1d array-like. If the output is a scalar, then it is converted into a 1d-array.
    
    tspan : 2-list or 2-tuple
        List or tuple containing the initial and final time values, the time span of the integration.
    
    y0 : 1d array_like
        The initial values. Scalar or 1d input. Scalar input is converted into a 1d-array.
        Complex integration is not supported. An error will be raised when passed complex values.
    
    h : integer or float, optional
        The initial stepsize to use.
    
    hmax : {np.inf or float}, optional
        Maximum stepsize. Default is np.inf which means no upper bound on the stepsize. If a finite number,
        it must be strictly greater than hmin and greater than zero.
    
    hmin : float, optional
        Minimum stepsize. If not given, then the stepsize has essentially no lower bound. The integration will be
        terminated and a ValueError raised when the stepsize h becomes very small i.e. smaller then 1000*EPS = 2.22e-13.
    
    order : {5, 8}, optional
        The order of the method to use. 5 corresponds to the RK5(4) pair while 8 uses the RK8(7) pair
    
    rtol, atol : float or array_like, optional
        Relative and absolute tolerances. If not given, then default tolerances depending on the order
        of the methods are used; 1e-4 and 1e-7 relative and absolute toleranes are used for 5th-order
        while 1e-7 and 1e-10 relative and absolute tolerances are used for 8th-order method.
    
    PI_controller : boolean, optional
        Enable or disable the use of a PI stepsize controller. PI controller is more robust for stiffer problems
        and results in fewer step rejections and a smoother solution than otherwise.
        On the other hand, non-PI controller is faster on the "easy" parts of the solution.
    
    dense_output : boolean, optional
        Whether to compute and output spline interpolants or not. The interpolants are computed "lazily" i.e.
        they're computed after the integration. If False, they're not computed at all.
    
    maxiter : int, optional
        Maximum number of iterations for adaptive integration of the ODEs. A warning statement is printed when reached.
        Default is one hundred thousand (10**5) iterations.
    
    args : {None, list or tuple}, optional
        List or tuple of additional parameters to be passed to the ODE function.
    
    
    Returns : Bunch object with the following fields defined:
        
        sol : The hermite spline interpolant.
        
        t : ndarray, shape (n_points,)
            Time points.
        
        y : ndarray, shape (n, n_points)
            Solution values at t.
        
        accepted_steps : integer
            The number of accepted steps
        
        rejected_steps : integer
            The number of rejected steps.
    
    
    References
    --------
    [1] FEHLBERG E., Low order classical Runge-Kutta formulas with step-size control and their application to some heat transfer problems, NASA TP, R-315, 1969, https://ntrs.nasa.gov/citations/19690021375
    [2] FEHLBERG E., Classical fifth, sixth, seventh and eighth order Runge-Kutta formulas with stepsize control, NASA TR R 287, Oct. 1968, https://ntrs.nasa.gov/citations/19680027281
    [3] J.R. Dormand, P.J. Prince, A family of embedded Runge-Kutta formulae, Journal of Computational and Applied Mathematics, Volume 6, Issue 1, 1980, Pages 19-26, ISSN 0377-0427, https://doi.org/10.1016/0771-050X(80)90013-3. (https://www.sciencedirect.com/science/article/pii/0771050X80900133)
    [4] L.F. Shampine, Global error estimation with one-step methods, Computers & Mathematics with Applications, Volume 12, Issue 7, Part A, 1986, Pages 885-894, ISSN 0898-1221, https://doi.org/10.1016/0898-1221(86)90032-5. (https://www.sciencedirect.com/science/article/pii/0898122186900325)
    [5] Jackson, K. R., et al. “A Theoretical Criterion for Comparing Runge-Kutta Formulas.” SIAM Journal on Numerical Analysis, vol. 15, no. 3, 1978, pp. 618–41. JSTOR, http://www.jstor.org/stable/2156590. Accessed 21 Jul. 2022.
    [6] GUSTAFSSON, K , LUNDH, M , AND SODERLIND, G. A PI stepslze control for the numerical solution of ordinary differential equations, BIT 28, 2 (1988), 270-287.
    [7] GUSTAFSSON,K. 1991. Control theoretic techniques for stepsize selection in explicit RungeKutta methods. ACM Trans. Math. Softw. 174 (Dec.) 533-554.

    """
    def __init__(self, fun, tspan, y0, h=None, hmax=np.inf, hmin=None, order=5, rtol=None, atol=None, PI_controller=True, dense_output=True, maxiter=10**5, args=None):
        
        y0, rtol, atol = self._validate_tol(y0, order, rtol, atol)
        
        tspan, h, hmax, hmin = self._validate_tspan(tspan, h, hmax, hmin)
        
        fun = self._validate_fun(fun, PI_controller, dense_output, args)
        
        if type(maxiter)!=int or maxiter<1:
            raise ValueError('maxiter must be a positive integer.')
        
        t, y, f, hacc, hrej = self._integrate(fun, tspan, y0, h, hmax, hmin, order, rtol, atol, PI_controller, maxiter)
        
        if dense_output:
            sol = self._spline_interpolant(fun, t, y, f, order)
        else:
            sol = None
        self.t = t
        self.y = y.T
        self.sol = sol
        self.accepted_steps = hacc
        self.rejected_steps = hrej
    
    def __repr__(self):
        attrs = ['t','y','sol','accepted_steps','rejected_steps']
        m = max(map(len,attrs)) + 1
        return '\n'.join([a.rjust(m) + ': ' + repr(getattr(self,a)) for a in attrs])
    
    
    @staticmethod
    def _rkf_tableau(order):
        if order==5:
            C = np.array([1/4, 3/8, 12/13, 1, 1/2])
            A = np.array([
                [1/4, *[0]*4],
                [3/32, 9/32, *[0]*3],
                [1932/2197, -7200/2197, 7296/2197, 0, 0],
                [439/216, -8, 3680/513, -845/4104, 0],
                [-8/27, 2, -3544/2565, 1859/4104, -11/40]
                ])
            B = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
            E = np.array([-1/360, 0, 128/4275, 2197/75240, -1/50, -2/55])
        else:
            C = np.array([2/27, 1/9, 1/6, 5/12, 1/2, 5/6, 1/6, 2/3, 1/3, 1, 0, 1])
            A = np.array([
                [2/27, *[0]*11],
                [1/36, 1/12, *[0]*10],
                [1/24, 0, 1/8, *[0]*9],
                [5/12, 0, -25/16, 25/16, *[0]*8],
                [1/20, 0, 0, 1/4, 1/5, *[0]*7],
                [-25/108, 0, 0, 125/108, -65/27, 125/54, *[0]*6],
                [31/300, 0, 0, 0, 61/225, -2/9, 13/900, *[0]*5],
                [2, 0, 0, -53/6, 704/45, -107/9, 67/90, 3, *[0]*4],
                [-91/108, 0, 0, 23/108, -976/135, 311/54, -19/60, 17/6, -1/12, *[0]*3],
                [2383/4100, 0, 0, -341/164, 4496/1025, -301/82, 2133/4100, 45/82, 45/164, 18/41, *[0]*2],
                [3/205, 0, 0, 0, 0, -6/41, -3/205, -3/41, 3/41, 6/41, 0, 0],
                [-1777/4100, 0, 0, -341/164, 4496/1025, -289/82, 2193/4100, 51/82, 33/164, 12/41, 0, 1]
                ])
            B = np.array([0, 0, 0, 0, 0, 34/105, 9/35, 9/35, 9/280, 9/280, 0, 41/840, 41/840])
            E = 41/840*np.array([1, *[0]*9, 1, -1, -1])
        return C, A, B, E
    
    
    def _rkf_step(self, fun, t_old, y_old, f_old, order, h):
        C, A, B, E = self._rkf_tableau(order)
        stages = A.shape[0]
        states = len(y_old)
        K = np.zeros(( stages+1, states ))
        K[0] = f_old
        for i in range(1,stages+1):
            K[i] = fun(t_old + C[i-1]*h,
                       y_old + h * A[i-1,:i] @ K[:i]
                       )
        y_new = y_old + h * B @ K
        t_new = t_old + h
        f_new = fun(t_new,y_new)
        err = h * E @ K
        return t_new, y_new, f_new, err
    
    
    @staticmethod
    def _scaled_error_norm(y_old, y_new, err, rtol, atol):
        sc = atol + rtol * np.maximum( np.abs(y_old), np.abs(y_new) )
        error_norm = RMS(err/sc)
        return error_norm
    
    
    @staticmethod
    def _compute_first_step(fun, t0, y0, f0, order, rtol, atol):
        scale = atol + np.abs(y0) * rtol
        d0 = RMS(y0 / scale)
        d1 = RMS(f0 / scale)
        if d0 < 1e-5 or d1 < 1e-5:
            h0 = 1e-6
        else:
            h0 = 0.01 * d0 / d1

        y1 = y0 + h0 * f0
        f1 = fun(t0 + h0, y1)
        d2 = RMS((f1 - f0) / scale) / h0

        if d1 <= 1e-15 and d2 <= 1e-15:
            h1 = max(1e-6, h0 * 1e-3)
        else:
            h1 = (0.01 / max(d1, d2)) ** (1 / order)

        return min(100 * h0, h1)
    
    
    def _integrate(self, fun, tspan, y0, h, hmax, hmin, order, rtol, atol, PI_controller, maxiter):
        MAX_FACTOR = 10.
        MIN_FACTOR = .1
        SAFETY = .9
        EXPONENT = -1/order
        KI = -.7/order
        KP = .4/order
        
        t0, tf = tspan
        y0 = np.atleast_1d(y0)
        f0 = fun(t0,y0)
        
        if h is None:
            h = self._compute_first_step(fun, t0, y0, f0, order, rtol, atol)
        
        if hmin is None:
            hmin = 10 * np.abs(np.nextafter(t0,tf) - t0)
        
        t = t0
        y = y0.copy()
        f = f0.copy()
        
        t_old = t0
        y_old = y0.copy()
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
                t_new, y_new, f_new, err = self._rkf_step(fun, t_old, y_old, f_old, order, h)
                t = np.hstack(( t, t_new ))
                y = np.vstack(( y, y_new ))
                f = np.vstack(( f, f_new ))
                break
            
            t_new, y_new, f_new, err = self._rkf_step(fun, t_old, y_old, f_old, order, h)
            error_norm = self._scaled_error_norm(y_old, y_new, err, rtol, atol)
            
            if error_norm < 1:
                if error_norm == 0:
                    FAC = MAX_FACTOR
                else:
                    if first_step or PI_controller==False:
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
                f = np.vstack(( f, f_new ))
                
                t_old = t_new
                y_old = y_new
                f_old = f_new
                
                hacc += 1
            else:
                h *= max(MIN_FACTOR, SAFETY * error_norm ** EXPONENT)
                step_rejected = True
                hrej += 1
        else:
            warn(f'Maximum number of iterations has been reached without successfully completing the integration of the interval {tspan} within the specified absolute and relative tolerances, {atol} and {rtol}.')
        
        return t, y, f, hacc, hrej
    
    
    def _spline_interpolant(self, fun, t, y, f, order):
        steps = t.size - 1
        h = t[1:] - t[:-1]
        k = y.shape[1]
        if order==5:
            y_mid = np.zeros((steps, k))
            f_mid = np.zeros_like(y_mid)
            d2y = np.zeros_like(f)
            for i in range(steps):
                _, y_mid[i], f_mid[i], _ = self._rkf_step(fun, t[i], y[i], f[i], order, .5*h[i])
                d2y[i] = (-46*y[i] + 32*y_mid[i] + 14*y[i+1]) / h[i]**2 - (12*f[i] + 16*f_mid[i] + 2*f[i+1]) / h[i]
            d2y[-1] = (14*y[-2] + 32*y_mid[-1] - 46*y[-1]) / h[-1]**2 + (2*f[-2] + 16*f_mid[-1] + 12*f[-1]) / h[-1]
            Y = np.zeros((t.size, 3, k))
            Y[:,0,:] = y
            Y[:,1,:] = f
            Y[:,2,:] = d2y
            sol = lambda x: BPoly.from_derivatives(t, Y)(x).T
        else:
            y13 = np.zeros(( steps, k ))
            y23 = np.zeros_like(y13)
            f13 = np.zeros_like(y13)
            f23 = np.zeros_like(y13)
            d2y = np.zeros_like(f)
            d3y = np.zeros_like(f)
            
            for i in range(steps):
                _, y13[i], f13[i], _ = self._rkf_step(fun, t[i], y[i], f[i], order, 1/3*h[i])
                _, y23[i], f23[i], _ = self._rkf_step(fun, t[i], y[i], f[i], order, 2/3*h[i])
                d2y[i] = (-291/2*y[i] + 0*y13[i] + 243/2*y23[i] + 24*y[i+1]) / h[i]**2 - (22*f[i] + 54*f13[i] + 27*f23[i] + 2*f[i+1]) / h[i]
                d3y[i] = (5073/2*y[i] + 1458*y13[i] - 6561/2*y23[i] - 714*y[i+1]) / h[i]**3 + (579/2*f[i] + 1296*f13[i] + 1539/2*f23[i] + 60*f[i+1]) / h[i]**2
            
            d2y[-1] = (24*y[-2] + 243/2*y13[-1] + 0*y23[-1] - 291/2*y[-1]) / h[-1]**2 + (2*f[-2] + 27*f13[-1] + 54*f23[-1] + 22*f[-1]) / h[-1]
            d3y[-1] = (714*y[-2] + 6561/2*y13[-1] - 1458*y23[-1] - 5073/2*y[-1]) / h[-1]**3 + (60*f[-2] + 1539/2*f13[-1] + 1296*f23[-1] + 579/2*f[-1]) / h[-1]**2
            
            Y = np.zeros((t.size, 4, k))
            Y[:,0,:] = y
            Y[:,1,:] = f
            Y[:,2,:] = d2y
            Y[:,3,:] = d3y
            sol = lambda x: BPoly.from_derivatives(t, Y)(x).T
        
        return sol
    
    
    @staticmethod
    def _validate_fun(fun, PI_controller, dense_output, args):
        if args is not None:
            if isinstance(args, tuple) or isinstance(args, list) and len(args)>0:
                fun = lambda t, y, fun=fun: np.atleast_1d( fun(t, y, *args) )
            else:
                raise ValueError('args must be a tuple or list with at least length of 1.')
        else:
            fun = lambda t, y, fun=fun: np.atleast_1d( fun(t, y) )
        
        if type(PI_controller)!=bool:
            raise TypeError('Argument PI_controller must be a boolean.')
        
        if type(dense_output)!=bool:
            raise TypeError('Argument dense_output must be a boolean.')
        
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
            if type(hmin)!=float and hmin>=hmax or hmin<=0:
                raise ValueError(f'If hmin is not {None}, it must be strictly less than hmax and strictly greater than zero.')
            else:
                pass
        else:
            pass
        
        if hmax!=np.inf and ( type(hmax)!=float or hmax<=hmin or hmax<=0 ):
            raise ValueError(f'If hmax is not {np.inf}, it must be strictly greater than hmin and zero.')
        
        return tspan, h, hmax, hmin
    
    
    @staticmethod
    def _validate_tol(y0, order, rtol, atol):
        y0 = np.atleast_1d(y0)
        if y0.ndim > 1:
            raise ValueError('y0 must be a vector (1d) of initial values.')
        elif y0.dtype not in [int,float]:
            raise TypeError('dtype of y0 must be int or float.')
        else:
            pass
        
        if order not in [5,8]:
            raise Exception('Value of order must be 5 or 8.')
        
        if rtol is None:
            rtol = {5:1e-4, 8:1e-7}[order]
        else:
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
        
        if atol is None:
            atol = {5:1e-7, 8:1e-10}[order]
        else:
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
        
        return y0, rtol, atol
