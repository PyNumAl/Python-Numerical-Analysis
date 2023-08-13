import numpy as np
from warnings import warn
from scipy.interpolate import CubicHermiteSpline as CHS

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
    
    This class implements Runge-Kutta-Fehlberg (RKF) pairs from orders 1 to 8 [1-2]. Practical results
    indicate that local extrapolation is preferable [3-5], and local extrapolation is performed except
    for the RK pairs RKF1(2), RKF2(3), and RKF3(4) because they are first-same-as-last (FSAL) when
    using the lower order formulas.
    
    For dense output, a "free" 3rd-order interpolant (scipy.interpolate.CubicHermiteSpline) is provided
    for a continuous solutions throughout the integration domain.
    
    A more robust PI stepsize control as described in [7-8] is implemented for adaptive stepsize control.
    
    It is important to note that Fehlberg's pairs fail in pure quadrature problems and their error-estimating
    strategies could lead to poor step size control should the leading error term not be dominant [9-10].
    
    
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
    
    order : {1, 2, 3, 4, 5, 6, 7, 8}, optional
        The order of the method to use:
            
            1 : RKF1(2) pair in Table XIV of [1]
            
            2 : RKF2(3) pair in Table XI of [1]
            
            3 : RKF3(4) pair in Table VIII Formula 2 of [1]
            
            4 (default) : RK4(5) pair in Table III Formula 2 of [1]
            
            5 : RKF5(6) pair in Table II of [2]
            
            6 : RKF6(7) pair in Table VIII of [2]
            
            7 : RKF7(8) in Table X of [2]
            
            8 : RKF8(9) in Table XII of [2]
                This pair and its coefficients are also implemented in
                (https://github.com/jacobwilliams/Fortran-Astrodynamics-Toolkit/blob/master/src/rk_module_variable_step.f90)
                
    
    rtol, atol : float or array_like, optional
        Relative and absolute tolerances.
    
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
            
        sol : scipy.interpolate.CubicHermiteSpline or scipy.interpolate.BPoly instance
            The hermite spline interpolant
            
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
    [6] J. M. (https://scicomp.stackexchange.com/users/127/j-m), Intermediate values (interpolation) after Runge-Kutta calculation, URL (version: 2013-05-24): https://scicomp.stackexchange.com/q/7363
    [7] GUSTAFSSON, K , LUNDH, M , AND SODERLIND, G. A PI stepslze control for the numerical solution of ordinary differential equations, BIT 28, 2 (1988), 270-287.
    [8] GUSTAFSSON,K. 1991. Control theoretic techniques for stepsize selection in explicit RungeKutta methods. ACM Trans. Math. Softw. 174 (Dec.) 533-554.
    [9] J.R. Dormand, M.E.A. El-Mikkwy and P.J. Prince, Higher order embedded Runge-Kutta-Nystrom formulae, IMA J. Numer. Anal. 8 (1987) 423-430.
    [10] DORMAND J. R. & PRINCE P.J. : "New Runge-Kutta algorithms for numerical simulation in dynamical astronomy", Celestial Mechanics, 18, 223-232, 1978.
    [11] Rackauckas, C. (https://scicomp.stackexchange.com/questions/14433/constructing-explicit-runge-kutta-methods-of-order-9-and-higher), Constructing explicit Runge Kutta methods of order 9 and higher, Jun 19, 2017 at 1:37
    [12] Rocha, A. (2018). Numerical Methods and Tolerance Analysis for Orbit Propagation, https://www.sjsu.edu/ae/docs/project-thesis/Angel.Rocha.Sp18.pdf
    """
    def __init__(self, fun, tspan, y0, h=None, hmax=np.inf, hmin=None, order=4, rtol=1e-3, atol=1e-6, maxiter=10**5, args=None):
        
        y0, rtol, atol = self._validate_tol(y0, order, rtol, atol)
        
        tspan, h, hmax, hmin = self._validate_tspan(tspan, h, hmax, hmin)
        
        fun = self._validate_fun(fun, args)
        
        if type(maxiter)!=int or maxiter<1:
            raise ValueError('maxiter must be a positive integer.')
        
        self.C, self.A, self.B, self.E, self.FSAL = self._rkf_tableau(order)
        t, y, f, hacc, hrej = self._integrate(fun, tspan, y0, h, hmax, hmin, order, rtol, atol, maxiter)
        
        sol = CHS(t, y, f)
        
        self.t = t
        self.y = y
        self.sol = sol
        self.hacc = hacc
        self.hrej = hrej
    
    def __repr__(self):
        attrs = ['t','y','sol','hacc','hrej']
        m = max(map(len,attrs)) + 1
        return '\n'.join([a.rjust(m) + ': ' + repr(getattr(self,a)) for a in attrs])    
    
    def _rkf_step(self, fun, t_old, y_old, f_old, order, h):
        C, A, B, E, FSAL = self.C, self.A, self.B, self.E, self.FSAL
        stages = A.shape[0]
        states = len(y_old)
        K = np.zeros(( stages, states ))
        K[0] = f_old
        if FSAL:
            for i in range(1,stages-1):
                K[i] = fun(t_old + C[i]*h,
                           y_old + h * A[i,:i] @ K[:i]
                           )
                y_new = y_old + h * B[:-1] @ K[:-1]
                t_new = t_old + h
                f_new = fun(t_new,y_new)
                K[-1] = f_new
        else:
            for i in range(1,stages):
                K[i] = fun(t_old + C[i]*h,
                           y_old + h * A[i,:i] @ K[:i]
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
        sc = atol + rtol * np.abs(y0)
        h0 = max(
            1e-6,
            RMS(.01 * sc) / RMS(
                np.maximum(
                    1e-8,
                    atol + rtol * np.abs(f0)
                    )
                )
            )
        y1 = y0 + h0*f0
        f1 = fun(t0 + h0, y1)
        fdot0 = (f1-f0)/h0
        TE = np.hstack(
            (
                RMS(
                    h0 ** (order+1) * 100 * fdot0 / np.maximum(1e-8, sc)
                    )
                ,
                RMS(h0*f0)
                )
            )
        if np.all(TE<=1e-12):
            h1 = max(1e-6, 1e-3*h0)
        else:
            h1 = h0 * TE.max() ** (-1/(1+order))
        return min(1e2*h0,h1)
    
    def _integrate(self, fun, tspan, y0, h, hmax, hmin, order, rtol, atol, maxiter):
        MAX_FACTOR = 10.
        MIN_FACTOR = .1
        SAFETY = .9
        EXPONENT = -1/(order+1)
        KI = -.7/(order+1)
        KP = .4/(order+1)
        
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
                hacc += 1
                break
            
            t_new, y_new, f_new, err = self._rkf_step(fun, t_old, y_old, f_old, order, h)
            error_norm = self._scaled_error_norm(y_old, y_new, err, rtol, atol)
            
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
    
    @staticmethod
    def _validate_fun(fun, args):
        if args is not None:
            if isinstance(args, tuple) or isinstance(args, list) and len(args)>0:
                fun = lambda t, y, fun=fun: np.atleast_1d( fun(t, y, *args) )
            else:
                raise ValueError('args must be a tuple or list with at least length of 1.')
        else:
            fun = lambda t, y, fun=fun: np.atleast_1d( fun(t, y) )
        
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
        
        if order not in [1,2,3,4,5,6,7,8]:
            raise Exception(f'Value of order must be in {[1,2,3,4,5,6,7,8]}')
        
        rtol = np.asarray(rtol)
        min_rtol = 1e3*EPS
        if rtol.ndim > 1:
            raise ValueError('rtol must be a scalar or vector (1d) of compatible shape.')
        elif rtol.ndim==1 and rtol.size!=y0.size:
            raise ValueError('If rtol is a vector, it must have the same size as y0.')
        elif rtol.dtype!=float:
            raise TypeError(f'dtype of rtol must be {float}')
        elif np.any(rtol<0):
            raise ValueError('rtol values must be positive.')
        elif np.any(rtol<min_rtol):
            warn(f'rtol value/s too small, setting to {min_rtol}')
            rtol = np.where(rtol<min_rtol, min_rtol, rtol)
        else:
            pass
        
        atol = np.asarray(atol)
        if atol.ndim > 1:
            raise ValueError('atol must be a scalar or vector (1d) of compatible shapes.')
        elif atol.ndim==1 and atol.size!=y0.size:
            raise ValueError('If atol is a vector, it must have the same size as y0.')
        elif atol.dtype!=float:
            raise TypeError(f'dtype of atol must be {float}')
        elif np.any(atol<0):
            raise ValueError('atol values must be nonnegative.')
        else:
            pass
        
        return y0, rtol, atol
    
    @staticmethod
    def _rkf_tableau(order):
        """
        B denotes the coefficients of the formula used to advance the integration
        while BCAP is the embedded method for error estimation. Fehlberg's pairs
        as implemented here don't perform local extrapolation. Hence, B and BCAP
        are the lower and higher-order formulas, respectively. Local extrapolation
        is not performed because Fehlberg's RK1(2), RK2(3), and RK3(4) pairs are FSAL
        when using the lower-order formula; and the RK8(9) pair doesn't give the higher
        9th-order pair, but only the lower, 8th-order pair and the error coefficients.
        """
        if order==1:
            FSAL = True
            C = np.array([0, 1/2, 1])
            A = np.array([
                [0, 0, 0],
                [1/2, 0, 0],
                [1/256, 255/256, 0]
                ])
            B = A[-1]
            # 2nd order coefficients
            BCAP = np.array([1/512, 255/256, 1/512])
        elif order==2:
            FSAL = True
            C = np.array([0, 1/4, 27/40, 1])
            A = np.zeros((4,4))
            A[1,:1] = [1/4]
            A[2,:2] = [-189/800, 729/800]
            A[3,:3] = [214/891, 1/33, 650/891]
            B = A[-1]
            # 3rd order coefficients
            BCAP = np.array([533/2106, 0, 800/1053, -1/78])
        elif order==3:
            FSAL = True
            C = np.array([0, 2/7, 7/15, 35/38, 1])
            A = np.zeros((5,5))
            A[1,:1] = [2/7]
            A[2,:2] = [77/900, 343/900]
            A[3,:3] = [805/1444, -77175/54872, 97125/54872]
            A[4,:4] = [79/490, 0, 2175/3626, 2166/9065]
            B = A[-1]
            # 4th order coefficients
            BCAP = np.array([229/1470, 0, 1125/1813, 13718/81585, 1/18])
        elif order==4:
            FSAL = False
            C = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
            A = np.zeros((6,6))
            A[1,:1] = [1/4]
            A[2,:2] = [3/32, 9/32]
            A[3,:3] = [1932/2197, -7200/2197, 7296/2197]
            A[4,:4] = [439/216, -8, 3680/513, -845/4104]
            A[5,:5] = [-8/27, 2, -3544/2565, 1859/4104, -11/40]
            # 4th order coefficients
            BCAP = np.array([25/216, 0, 1408/2565, 2197/4104, -1/5, 0])
            # 5th order coefficients
            B = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
        elif order==5:
            FSAL = False
            C = np.array([0, 1/6, 4/15, 2/3, 4/5, 1, 0, 1])
            A = np.zeros((8,8))
            A[1,:1] = [1/6]
            A[2,:2] = [4/75, 16/75]
            A[3,:3] = [5/6, -8/3, 5/2]
            A[4,:4] = [-8/5, 144/25, -4, 16/25]
            A[5,:5] = [361/320, -18/5, 407/128, -11/80, 55/128]
            A[6,:6] = [-11/640, 0, 11/256, -11/160, 11/256, 0]
            A[7,:7] = [93/640, -18/5, 803/256, -11/160, 99/256, 0, 1]
            # 5th order coefficients
            BCAP = np.array([31/384, 0, 1125/2816, 9/32, 125/768, 5/66, 0, 0])
            # 6th order coefficients
            B = np.array([7/1408, 0, 1125/2816, 9/32, 125/768, 0, 5/66, 5/66])
        elif order==6:
            FSAL = False
            C = np.array([0, 2/33, 4/33, 2/11, 1/2, 2/3, 6/7, 1, 0, 1])
            A = np.zeros((10,10))
            A[1,:1] = [2/33]
            A[2,:2] = [0, 4/33]
            A[3,:3] = [1/22, 0, 3/22]
            A[4,:4] = [43/64, 0, -165/64, 77/32]
            A[5,:5] = [-2383/486, 0, 1067/54, -26312/1701, 2176/1701]
            A[6,:6] = [10077/4802, 0, -5643/686, 116259/16807, -6240/16807, 1053/2401]
            A[7,:7] = [-733/176, 0, 141/8, -335763/23296, 216/77, -4617/2816, 7203/9152]
            A[8,:8] = [15/352, 0, 0, -5445/46592, 18/77, -1215/5632, 1029/18304, 0]
            A[9,:9] = [-1833/352, 0, 141/8, -51237/3584, 18/7, -729/512, 1029/1408, 0, 1]
            # 6th order coefficients
            BCAP = np.array([
                77/1440, 0, 0, 1771561/6289920, 32/105, 243/2560,
                16807/74880, 11/270, 0, 0
                ])
            # 7th order coefficients
            B = np.array([
                11/864, 0, 0, 1771561/6289920, 32/105, 243/2560,
                16807/74880, 0, 11/270, 11/270
                ])
        elif order==7:
            FSAL = False
            C = np.array([0, 2/27, 1/9, 1/6, 5/12, 1/2, 5/6, 1/6, 2/3, 1/3, 1, 0, 1])
            A = np.zeros((13,13))
            A[1,:1] = [2/27]
            A[2,:2] = [1/36, 1/12]
            A[3,:3] = [1/24, 0, 1/8]
            A[4,:4] = [5/12, 0, -25/16, 25/16]
            A[5,:5] = [1/20, 0, 0, 1/4, 1/5]
            A[6,:6] = [-25/108, 0, 0, 125/108, -65/27, 125/54]
            A[7,:7] = [31/300, 0, 0, 0, 61/225, -2/9, 13/900]
            A[8,:8] = [2, 0, 0, -53/6, 704/45, -107/9, 67/90, 3]
            A[9,:9] = [-91/108, 0, 0, 23/108, -976/135, 311/54,
                       -19/60, 17/6, -1/12]
            A[10,:10] = [2383/4100, 0, 0, -341/164, 4496/1025,
                         -301/82, 2133/4100, 45/82, 45/164, 18/41]
            A[11,:11] = [3/205, 0, 0, 0, 0, -6/41, -3/205, -3/41, 3/41, 6/41, 0]
            A[12,:12] = [-1777/4100, 0, 0, -341/164, 4496/1025,
                         -289/82, 2193/4100, 51/82, 33/164, 12/41, 0, 1]
            # 7th order coefficients
            BCAP = np.array([
                41/840, 0, 0, 0, 0, 34/105, 9/35, 9/35,
                9/280, 9/280, 41/840, 0, 0
                ])
            # 8th order coefficients
            B = np.array([
                0, 0, 0, 0, 0, 34/105, 9/35, 9/35,
                9/280, 9/280, 0, 41/840, 41/840
                ])
        elif order==8:
            FSAL = False
            C = np.array([
                0,
                .44368940376498183109599404281370,
                .66553410564747274664399106422055,
                .99830115847120911996598659633083,
                .3155,
                .50544100948169068626516126737384,
                .17142857142857142857142857142857,
                .82857142857142857142857142857143,
                .66543966121011562534953769255586,
                .24878317968062652069722274560771,
                .109,
                .891,
                .3995,
                .6005,
                1,
                0,
                1
                ])
            A = np.zeros((17,17))
            A[1,:1] = [.44368940376498183109599404281370]
            A[2,:2] = [
                .16638352641186818666099776605514,
                .49915057923560455998299329816541,
                ]
            A[3,:3] = [
                .24957528961780227999149664908271,
                0,
                .74872586885340683997448994724812
                ]
            A[4,:4] = [
                .20661891163400602426556710393185,
                0,
                .17707880377986347040380997288319,
                -.68197715413869494669377076815048e-1
                ]
            A[5,:5] = [
                .10927823152666408227903890926157,
                0,
                0,
                .40215962642367995421990563690087e-2,
                .39214118169078980444392330174325
                ]
            A[6,:6] = [
                .98899281409164665304844765434355e-1,
                0,
                0,
                .35138370227963966951204487356703e-2,
                .12476099983160016621520625872489,
                -.55745546834989799643742901466348e-1
                ]
            A[7,:7] = [
                -.36806865286242203724153101080691,
                0,
                0,
                0,
                -.22273897469476007645024020944166e+1,
                .13742908256702910729565691245744e+1,
                .20497390027111603002159354092206e+1
                ]
            A[8,:8] = [
                .45467962641347150077351950603349e-1,
                0, 0, 0, 0,
                .32542131701589147114677469648853,
                .28476660138527908888182420573687,
                .97837801675979152435868397271099e-2
                ]
            A[9,:9] = [
                .60842071062622057051094145205182e-1,
                0, 0, 0, 0,
                -.21184565744037007526325275251206e-1,
                .19596557266170831957464490662983,
                -.42742640364817603675144835342899e-2,
                .17434365736814911965323452558189e-1
                ]
            A[10,:10] = [
                .54059783296931917365785724111182e-1,
                0, 0, 0, 0, 0,
                .11029825597828926530283127648228,
                -.12565008520072556414147763782250e-2,
                .36790043477581460136384043566339e-2,
                -.57780542770972073040840628571866e-1,
                ]
            A[11,:11] = [
                .12732477068667114646645181799160,
                0, 0, 0, 0, 0, 0,
                .11448805006396105323658875721817,
                .28773020709697992776202201849198,
                .50945379459611363153735885079465,
                -.14799682244372575900242144449640
                ]
            A[12,:12] = [
                -.36526793876616740535848544394333e-2,
                0, 0, 0, 0,
                .81629896012318919777819421247030e-1,
                -.38607735635693506490517694343215,
                .30862242924605106450474166025206e-1,
                -.58077254528320602815829374733518e-1,
                .33598659328884971493143451362322,
                .41066880401949958613549622786417,
                -.11840245972355985520633156154536e-1
                ]
            A[13,:13] = [
                -.12375357921245143254979096135669e+1,
                0, 0, 0, 0,
                -.24430768551354785358734861366763e+2,
                .54779568932778656050436528991173,
                -.44413863533413246374959896569346e+1,
                .10013104813713266094792617851022e+2,
                -.14995773102051758447170985073142e+2,
                .58946948523217073620824539651427e+1,
                .17380377503428984877616857440542e+1,
                .27512330693166730263758622860276e+2
                ]
            A[14,:14] = [
                -.35260859388334522700502958875588,
                0, 0, 0, 0,
                -.18396103144848270375044198988231,
                -.65570189449741645138006879985251,
                -.39086144880439863435025520241310,
                .26794646712850022936584423271209,
                -.10383022991382490865769858507427e+1,
                .16672327324258671664727346168501e+1,
                .49551925855315977067732967071441,
                .11394001132397063228586738141784e+1,
                .51336696424658613688199097191534e-1
                ]
            A[15,:15] = [
                .10464847340614810391873002406755e-2,
                0, 0, 0, 0, 0, 0, 0,
                -.67163886844990282237778446178020e-2,
                .81828762189425021265330065248999e-2,
                -.42640342864483347277142138087561e-2,
                .28009029474168936545976331153703e-3,
                -.87835333876238676639057813145633e-2,
                .10254505110825558084217769664009e-1,
                0
                ]
            A[16,:16] = [
                -.13536550786174067080442168889966e+1,
                0, 0, 0, 0,
                -.18396103144848270375044198988231,
                -.65570189449741645138006879985251,
                -.39086144880439863435025520241310,
                .27466285581299925758962207732989,
                -.10464851753571915887035188572676e+1,
                .16714967667123155012004488306588e+1,
                .49523916825841808131186990740287,
                .11481836466273301905225795954930e+1,
                .41082191313833055603981327527525e-1,
                0,
                1
                ]
            # 9th order coefficients
            B = np.zeros(17)
            B[0] = 0.0015295880243556095072445954381230

            B[8] = 0.25983725283715403018887023171963
            B[9] = 0.092847805996577027788063714302190
            B[10] = 0.16452339514764342891647731842800
            B[11] = 0.17665951637860074367084298397547
            B[12] = 0.23920102320352759374108933320941
            B[13] = 0.0039484274604202853746752118829325
            
            B[15] = 0.030726495475860640406368305522124
            B[16] = 0.030726495475860640406368305522124
            # 8th order coefficients
            BCAP = np.zeros(17)
            BCAP[0] = 0.032256083500216249913612900960247
            BCAP[8] = 0.25983725283715403018887023171963
            BCAP[9] = 0.092847805996577027788063714302190
            BCAP[10] = 0.16452339514764342891647731842800
            BCAP[11] = 0.17665951637860074367084298397547
            BCAP[12] = 0.23920102320352759374108933320941
            BCAP[13] = 0.0039484274604202853746752118829325
            BCAP[14] = 0.030726495475860640406368305522124
            
        E =  B - BCAP
        
        return C, A, B, E, FSAL
