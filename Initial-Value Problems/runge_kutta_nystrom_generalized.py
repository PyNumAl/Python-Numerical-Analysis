import numpy as np
from numpy.linalg import norm
from warnings import warn
from scipy.interpolate import BPoly
EPS = np.finfo(float).eps

def RMS(x,axis):
    out = np.sqrt(
        np.mean(
            np.square(x)
            ,
            axis
            )
        )
    return out

class RKNG:
    """
    Solve general second-order scalar/vector initial-value problems of the form:
        
        d2y / dt2 = f(t, y, yp)
        
    
    This class implements several Runge-Kutta-Nystrom-Generalized (RKNG) pairs
    for solving general second-order ordinary differential equations. RKNG methods
    are often more efficient at solving 2nd-order ODEs than their Runge-Kutta (RK)
    counterparts solving the same 2nd-order ODE transformed to its 1st-order ODE
    equivalent. If the 2nd-order ODE being solved does not depend on the
    1st derivative, y', then specially designed RK methods, known as 
    Runge-Kutta-Nystrom (RKN) methods in literature, are more efficient than RKNG methods
    and require fewer function evaluations to achieve the same order of accuracy
    as their RKNG counterparts.
    
    A robust PI stepsize control as described in [6-7] is chosen and implemented for
    adaptive stepsize control.
    
    
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
    
    method : string, optional
        Integration method to use:
            
            'RKNG4': Fourth-Order Runge-Kutta-Nystrom method [10-12].
                Originally a four-stage, fixed-step fourth-order method,
                adaptivity and error estimation (y component only) is achieved
                as described in [10] without additional stages. The error
                estimate is assumed to be of order 3, essentially turning
                the method into a 4-stage RKNG4(3) method.
            
            'RKNG43': Runge-Kutta-Nystrom-Generalized method of order 4(3) [1].
                The error of both y and y' components is
                controlled assuming accuracy of the third-order methods,
                but steps are taken using the fourth-order accurate
                formulas (local extrapolation is done). It does not have
                the First-Same-As-Last (FSAL) property.
            
            'RKNG54' (default): Runge-Kutta-Nystrom-Generalized method of order 5(4) [1].
                The error of both y and y' components is 
                controlled assuming accuracy of the fourth-order methods,
                but steps are taken using the fifth-order accurate
                formulas (local extrapolation is done). It has the FSAL property.
            
            'RKNG5456': Runge-Kutta-Nystrom-Generalized method of order 5(4) [13].
                This RKNG pair has fewer y stages than y' stages, 5 and 6
                respectively. This is more efficient for problems that have
                low-y' dependency. The underlying RK method of this RKNG pair
                is the RK54 pair in [14]. Hence, for 2nd order ODEs that depend
                only on y' and t, i.e. d2y/dt2 = f(t,y'), then this will perform exactly
                the same as that RK54 pair. Local extrapolation is performed on each
                step and the pair is FSAL.
            
            'RKNG5459': Runge-Kutta-Nystrom-Generalized method of order 5(4) [13].
                This RKNG pair has fewer y stages than y' stages, 5 and 9
                respectively. This is more efficient for problems that have
                low-y' dependency. Its error estimator is more robust than the RKNG5456
                pair due to having 9 total 2nd derivative stages than for the aforementioned's
                6 stages. Local extrapolation is performed on each step and the pair is FSAL.
            
            'RKNG6459': Runge-Kutta-Nystrom-Generalized method of order 6(4) [13].
                This RKNG pair has fewer y stages than y' stages, 5 and 9
                respectively. This is more efficient for problems that have
                low-y' dependency. Its underlying RKN method is the RKN6(4) method
                from [15]. Hence, for 2nd order ODEs that have low or no y'-dependency,
                it will perform almost the same or exactly as that RKN6(4) pair respectively.
                Local extrapolation is performed on each step and the pair is FSAL.
            
            'RKNG56': Runge-Kutta-Nystrom-Generalized method of order 5(6) [2-3].
                The error (only the y component) is controlled
                assuming accuracy of the fifth-order methods. Local
                extrapolation is not done i.e. steps are taken using the
                lower, 5th-order accurate formula because the Butcher tableau
                of this RKNG method is FSAL when using the 5th-order method.
                
                Fehlberg's RKNG pairs (including this) may produce large global
                errors on some problems since the local error of the
                y' components are not controlled [8-9].
            
            'RKNG65': Runge-Kutta-Nystrom-Generalized method of order 6(5) [8].
                Eight-stage (5,6) RKNG pair that controls both the local error
                in y and y' and performs local extrapolation in each step.
                It does not have the FSAL property.
                
                IMPORTANTE NOTE: There is a printing error in [8] - the A[7,3]-th coefficient
                is lacking a zero in the denominator; it is printed as -31680158501/747419400
                which evaluates to -42.38605326674689. The coefficients would not sum to 0.5
                as a result. Adding another zero in the denominator, it becomes
                -31680158501/7474194000, which evaluates to -4.238605326674689. This allows
                the A[7]-th row to sum to 0.5.
            
            'RKNG67': Runge-Kutta-Nystrom-Generalized method of order 6(7) [2-3].
                The error (only the y component) is controlled assuming accuracy
                of the sixth-order methods. Local extrapolation is not done i.e.
                steps are taken using the lower, 6th-order accurate formula because
                the Butcher tableau of this RKNG method is FSAL when using the
                6th-order method.
                
                Fehlberg's RKNG pairs (including this) may produce large global
                errors on some problems since the local error of the
                y' components are not controlled [8-9].
            
            'RKNG78': Runge-Kutta-Nystrom-Generalized method of order 7(8) [2-3].
                The error (only the y component) is controlled assuming accuracy
                of the seventh-order methods. Local extrapolation is not done i.e.
                steps are taken using the lower, 7th-order accurate formula because
                the Butcher tableau of this RKNG method is FSAL when using the
                7th-order method.
                
                Fehlberg's RKNG pairs (including this) may produce large global
                errors on some problems since the local error of the
                y' components are not controlled [8-9].
        
    interpolant : {5,8}, optional
        Order of the interpolant to use:
            
            5 (default) : "Free" quintic hermite spline interpolant.
                This interpolant is relatively sufficient for the RKNG
                pairs that have error estimators of order 4 or less [4].
            
            8 : 8th-order hermite spline interpolant
                Interpolant for the higher order RKNG pairs. This is obtained by 
                calculating the solution, velocity, and 2nd derivative values at
                the midpoint of each subinterval after the integration is done.
                With the solution, velocity, and 2nd derivative values
                known at the left and right endpoints and midpoint of each subinterval,
                an 8th-degree spline may now be constructed using divided differences.
                
                Symbolic Python (SymPy) was used to obtain the expressions for the
                3rd and 4th derivatives of the 8th-degree spline at the left and right endpoints.
                scipy.interpolate.BPoly.from_derivatives is used to construct the high-order
                spline interpolant.The constructed spline interpolant is actually a 9th-degree
                hermite spline because derivatives up to the 4th derivative are provided in 
                order to not "lose information" from the underlying, theoretical 
                (and not constructed) 8th-degree spline.
    
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
            Hermite spline interpolant
            
        hacc : integer
            The number of accepted steps
            
        hrej : integer
            The number of rejected steps.
    
    
    References
    ---------
    [1] Fine, J.M. Low order practical Runge-Kutta-Nyström methods. Computing 38, 281–297 (1987). https://doi.org/10.1007/BF02278707
    [2] Fehlberg, E. Klassische Runge-Kutta-Nyström-Formeln mit Schrittweiten-Kontrolle für Differentialgleichungenx¨=f(t,x,x˙) . Computing 14, 371–387 (1975). https://doi.org/10.1007/BF02253548
    [3] Fehlberg, E. Classical seventh-, sixth-, and fifth-order Runge-Kutta-Nystrom formulas with stepsize control for general second-order differential equations . NASA-TR-R-432, M-546, NASA Marshall Space Flight Center Huntsville, AL, United States (1974). https://ntrs.nasa.gov/citations/19740026877
    [4] J. M. (https://scicomp.stackexchange.com/users/127/j-m), Intermediate values (interpolation) after Runge-Kutta calculation, URL (version: 2013-05-24): https://scicomp.stackexchange.com/q/7363
    [5] Hermite interpolation. https://en.wikipedia.org/wiki/Hermite_interpolation
    [6] GUSTAFSSON, K , LUNDH, M , AND SODERLIND, G. A PI stepslze control for the numerical solution of ordinary differential equations, BIT 28, 2 (1988), 270-287.
    [7] GUSTAFSSON,K. 1991. Control theoretic techniques for stepsize selection in explicit RungeKutta methods. ACM Trans. Math. Softw. 174 (Dec.) 533-554.
    [8] P.W. Sharp, J.M. Fine, Some Nyström pairs for the general second-order initial-value problem, Journal of Computational and Applied Mathematics, Volume 42, Issue 3, 1992, Pages 279-291, ISSN 0377-0427, https://doi.org/10.1016/0377-0427(92)90081-8. (https://www.sciencedirect.com/science/article/pii/0377042792900818)
    [9] J.R. Dormand, M.E.A. El-Mikkwy and P.J. Prince, Higher order embedded Runge-Kutta-Nystrom formulae, IMA J. Numer. Anal. 8 (1987) 423-430.
    [10] E Lund et al 2009 JINST 4 P04001, Track parameter propagation through the application of a new adaptive Runge-Kutta-Nyström method in the ATLAS experiment, https://iopscience.iop.org/article/10.1088/1748-0221/4/04/P04001
    [11] E.J. Nystrom, Uber die numerische Integration von Differentialgleichungen, Acta Soc. Sci. Fenn. 50 (1925) 1.
    [12] E. Hairer, S. P. Nørsett, and G. Wanner, Solving Ordinary Differential Equations I: Nonstiff Problems, vol. 8 of Springer Series in Computational Mathematics, Springer, Berlin, Germany, 1993.
    [13] A. Murua, Runge-Kutta-Nyström methods for general second order ODEs with application to multi-body systems, Applied Numerical Mathematics, Volume 28, Issues 2–4, 1998, Pages 387-399, ISSN 0168-9274, https://doi.org/10.1016/S0168-9274(98)00055-5. (https://www.sciencedirect.com/science/article/pii/S0168927498000555)
    [14] J.R. Dormand and P.J. Prince, A family of embedded Runge-Kutta formulae, J. Comput. Appl. Math. 6 (1980) 19-26.
    [15] J.R. Dormand, M.E.A. El-Mikkawy and PJ. Prince, Families of Runge-Kutta-Nyström formulae, IMA J. Nummer Anal. 7 (1987) 235-250.
    """
    def __init__(self, fun, tspan, y0, yp0, method='RKNG54', interpolant=5, h=None, hmax=np.inf, hmin=0., rtol=1e-3, atol=1e-6, maxiter=10**5, args=None):
        fun = self._validate_fun(fun, args)
        tspan, h, hmax, hmin = self._validate_tspan(tspan, h, hmax, hmin)
        y0, yp0, rtol, atol = self._validate_tol(y0, yp0, rtol, atol)
        self._validate_misc(maxiter, method, interpolant)
        
        self.C, self.A, self.AP, self.B, self.E, self.FSAL = self._rkng_tableau(method)
        t, y, yp, f, hacc, hrej = self._integrate(fun, tspan, y0, yp0, method, h, hmax, hmin, rtol, atol, maxiter)
        sol = self._spline_interpolant(fun, t, y, yp, f, rtol, atol, method, interpolant)
        self.t = t
        self.y = y
        self.sol = sol
        self.hacc = hacc
        self.hrej = hrej
    
    
    def __repr__(self):
        attrs = ['t','y','sol','hacc','hrej']
        m = max(map(len,attrs)) + 1
        return '\n'.join([a.rjust(m) + ': ' + repr(getattr(self,a)) for a in attrs])
    
    
    @staticmethod
    def _compute_first_step(fun, t0, y0, yp0, f0, p, rtol, atol):
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
            h1 = (.01 / max(norm1, norm2)) ** (1/(p+1))
        return min(100 * h0, h1)
    
    
    @staticmethod
    def _rkng_tableau(method):
        if method=='RKNG4':
            FSAL = False
            C = np.array([0, .5, .5, 1.])
            A = np.array([
                [0, 0, 0, 0],
                [1/8, 0, 0, 0],
                [1/8, 0, 0, 0],
                [0, 0, 1/2, 0]
                ])
            AP = np.array([
                [0, 0, 0, 0],
                [1/2, 0, 0, 0],
                [0, 1/2, 0, 0],
                [0, 0, 1., 0]
                ])
            B = np.array([
                [1/6, 1/6, 1/6, 0],
                [1/6, 1/3, 1/3, 1/6]
                ])
            E = np.array([1, -1, -1, 1])
        elif method=='RKNG43':
            FSAL = False
            C = np.array([0, 2/9, 1/3, 3/4, 1])
            A = np.array([
                [0, 0, 0, 0, 0],
                [2/81, 0, 0, 0, 0],
                [1/36, 1/36, 0, 0, 0],
                [9/128, 0, 27/128, 0, 0],
                [11/60, -3/20, 9/25, 8/75, 0]
                ])
            AP = np.array([
                [0, 0, 0, 0, 0],
                [2/9, 0, 0, 0, 0],
                [1/12, 1/4, 0 , 0, 0],
                [69/128, -243/128, 135/64, 0, 0],
                [-17/12, 27/4, -27/5, 16/15, 0]
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
        elif method=='RKNG54':
            FSAL = True
            C = np.array([0, 8/39, 4/13, 5/6, 43/47, 1, 1])
            A = np.zeros((7,7))
            A[1,:1] = [32/1521]
            A[2,:2] = [4/169, 4/169]
            A[3,:3] = [175/5184, 0, 1625/5184]
            A[4,:4] = [-342497279/5618900760, 6827067/46824173, 35048741/102161832, -2201514/234120865]
            A[5,:5] = [-7079/52152, 767/2173, 14027/52152, 30/2173, 0]
            A[6,:6] = [4817/51600, 0, 388869/1216880, 3276/23575, -1142053/22015140, 0]
            AP = np.zeros((7,7))
            AP[1,:1] = [8/39]
            AP[2,:2] = [1/13, 3/13]
            AP[3,:3] = [7385/6912, -9425/2304, 13325/3456]
            AP[4,:4] = [223324757/91364240, -174255393/18272848, 382840094/46824173, -39627252/234120865]
            AP[5,:5] = [108475/36464, -9633/848, 7624604/806183, 8100/49979, -4568212/19446707]
            AP[6,:6] = [4817/51600, 0, 1685099/3650640, 19656/23575, -53676491/88060560, 53/240]
            B = np.array([
                [4817/51600, 0, 388869/1216880, 3276/23575, -1142053/22015140, 0, 0],
                [4817/51600, 0, 1685099/3650640, 19656/23575, -53676491/88060560, 53/240, 0]
                ])
            E = np.array([
                [8151/2633750, 0, -1377519/186334750, 586872/28879375, -36011118/2247378875, 0, 0],
                [8151/2633750, 0, -5969249/559004250, 3521232/28879375, -846261273/4494757750, 4187/36750, -1/25]
                ])
        elif method=='RKNG5456':
            FSAL = True
            C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
            A = np.zeros((7,7))
            A[1,:1] = [1/50]
            A[2,:2] = [9/400, 9/400]
            A[3,:3] = [26/225, -6/25, 4/9]
            A[4,:4] = [65536/413343, -6940/15309, 284504/413343, 212/137781]
            A[5,:5] = [35/384, 0, 50/159, 25/192, -243/6784]
            A[6,:6] = [35/384, 0, 50/159, 25/192, -243/6784, 0]
            AP = np.zeros((7,7))
            AP[1,:1] = [1/5]
            AP[2,:2] = [3/40, 9/40]
            AP[3,:3] = [44/45, -56/15, 32/9]
            AP[4,:4] = [19372/6561, -25360/2187, 64448/6561, -212/729]
            AP[5,:5] = [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
            AP[6,:6] = [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84]
            B = np.array([
                [35/384, 0, 50/159, 25/192, -243/6784, 0, 0],
                [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0]
                ])
            E = np.array([
                [-71/57600, 0, 71/23850, -71/9600, 1917/339200, 0, 0],
                [-71/57600, 0, 71/16695, -71/1920, 17253/339200, -22/525, 1/40]
                ])
        elif method=='RKNG5459':
            FSAL = True
            C = np.array([
                0, 771/3847, 771/3847, 3051/6788, 4331/6516,
                4331/6516, 10463/11400, 10463/11400, 1., 1.
                ])
            A = np.zeros((10,10))
            A[[1,2],0] = 594441/29598818
            A[3,:3] = [-311625081/28869248936, 128/8219,
                       1015645542/10554116159]
            A[[4,5],:4] = [1852480471/26299626569, -247/14069,
                           648800762/5897141541, 519849979/8963946221]
            A[[6,7],:6] = [229929851/7158517178, 113395809/8665398238,
                           4865737279/19748497543, 340133672/10137556453,
                           738/11587, 509108839/15737542787]
            A[[8,9],:8] = [164505448/2653157365, 0, 9357192/40412735,
                           736403089/7677655029, 960089/17896194,
                           482653907/11393392643, -47281957/150822000,
                           6715245221/20471724521]
            AP = np.zeros_like(A)
            AP[1,:1] = [771/3847]
            AP[2,:2] = [771/7694]*2
            AP[3,:3] = [-264272222/4845505509, -9458865980/12714902623,
                        17133165419/13729279360]
            AP[4,:4] = [1943604853/18116134489, -2470367896/7636570485,
                        1733951147/3918733571, 4613437932/10523350595]
            AP[5,:5] = [369952551/2046485744, 281630106828/143708239525,
                        -9868262031/5606899429, 208606720/5597531799,
                        792516107/3209667255]
            AP[6,:6] = [
                -2089737154/15083636501, -39924138556/8175090533,
                72922890855/14010113917, 9484193351/15493195043,
                -17895265139/12412283353, 278232/177835
                ]
            AP[7,:7] = [
                -1762013041/13188190032, -22636373880/4795132451,
                30527401913/6048941340, 11564353310/19632283007,
                -50677425731/36595197965, 12408/8167, 10722067/5782709432
                ]
            AP[8,:8] = [
                8034174097/12261534992, 72032427203/6782716235,
                -90566218637/8185393121, 18770105843/41171085325,
                28010344030/6199889941, -21917292279/4540377286,
                -236637914115/8183370127, 71217630373/2409299224
                ]
            AP[9,:9] = [
                164505448/2653157365, 0 , 3042/10505, 1586146904/9104113535,
                4394/27465, 2081836558/16479128289, -50461/13230,
                13928550541/3490062596, 91464477/8242174145
                ]
            B = np.zeros((2,10))
            B[0,:8] = [164505448/2653157365, 0, 9357192/40412735,
                           736403089/7677655029, 960089/17896194,
                           482653907/11393392643, -47281957/150822000,
                           6715245221/20471724521]
            B[1,:9] = [
                164505448/2653157365, 0 , 3042/10505, 1586146904/9104113535,
                4394/27465, 2081836558/16479128289, -50461/13230,
                13928550541/3490062596, 91464477/8242174145
                ]
            # Embedded lower order approximation
            BCAP = np.zeros_like(B)
            BCAP[0,:8] = [
                53757362/127184461, 0, -426604134200/784970228543,
                605250622521/1277181566164, 79190349755/672734111688,
                2185/13032, 1075258194511/113794252800, -2244129055937/234065187600
                ]
            BCAP[1,:10] = [
                53757362/127184461, 0, -138687950/204047369,
                161961633/188152853, 36242723/103243418, 1/2,
                1147554103/9981952, -2395015001/20532034, 1, 23/100
                ]
            E = BCAP - B
        elif method=='RKNG6459':
            FSAL = True
            C = np.array([0, .1, .1, .3, .7, .7, .68, .68, 1., 1.])
            A = np.zeros((10,10))
            A[1,:1] = [1/200]
            A[2,:2] = [1/200, 0]
            A[3,:3] = [-1/2200, 0, 1/22]
            A[4,:4] = [637/6600, 0, -7/110, 7/33]
            A[5,:5] = [637/6600, 0, -7/110, 7/33, 0]
            A[6,:6] = [225437/1968750, 0, -30073/281250, 65569/281250,
                -5534276509/402334453125, 11940527048/2816341171875]
            A[7,:7] = [225437/1968750, 0, -30073/281250, 65569/281250,
                -5534276509/402334453125, 11940527048/2816341171875, 0]
            A[8,:8] = [151/2142, 0, 5/116, 385/1368, 6/55, 2017/9240, -8/35, 6058/983535]
            A[9,:9] = [151/2142, 0, 5/116, 385/1368, 6/55, 2017/9240, -8/35, 6058/983535, 0]
            AP = np.zeros_like(A)
            AP[1,:1] = [.1]
            AP[2,:2] = [1/20, 1/20]
            AP[3,:3] = [-3/20, -27/20, 9/5]
            AP[4,:4] = [596/495, -4/11, -427/264, 5849/3960]
            AP[5,:5] = [-340183643/75879540, -2893317/88748, 50178653/1264659,
                        -40900531/18969885, 5/19]
            AP[6,:6] = [-396583/95400, -169371/5000, 21382771/530000, -1735451/954000,
                        1/6, 1/53]
            AP[7,:7] = [-7118348801/1589316300, -169371/5000, 4295617784819/105954420000,
                        -160583069351/127145304000, -1292153/2665520,
                        3399687081/565090240, -17500/3029]
            AP[8,:8] = [2243113/2686068, 245/24, -19392535/1745568, 165115/361152,
                        193763/351120, 34289/24640, -461732/327845, 24232/327845]
            AP[9,:9] = [151/2142, 0, 25/522, 275/684, 4/11, 2017/2772, -5/7, 15145/786828, 1/12]
            B = np.array([
                [151/2142, 0, 5/116, 385/1368, 6/55, 2017/9240, -8/35, 6058/983535, 0, 0],
                [151/2142, 0, 25/522, 275/684, 4/11, 2017/2772, -5/7, 15145/786828, 1/12, 0]
                ])
            E = np.array([
                [-165817/2677500, 0, 165817/1450000, -1160719/17100000, 13760344867/1072891875000, -2757339150101/30040972500000, 8/35, -3298762/24588375, 0, 0],
                [-165817/2677500, 0, 165817/1305000, -165817/1710000, 13760344867/321867562500, -2757339150101/9012291750000, 5/7, -1649381/3934140, -1/100, 1/100]
                ])
        elif method=='RKNG56':
            FSAL = True
            C = np.array([0, 4/15, 2/5, 3/5, 9/10, 3/4, 2/7, 1, 1])
            A = np.zeros((9,9))
            A[1,:1] = [8/225]
            A[2,:2] = [1/25,1/25]
            A[3,:3] = [9/160, 81/800, 9/400]
            A[4,:4] = [81/640, 0, 729/3200, 81/1600]
            A[5,:5] = [11283/88064, 0, 3159/88064, 7275/44032, -33/688]
            A[6,:6] = [6250/194481, 0, 0, 0, -3400/194481, 1696/64827]
            A[7,:7] = [-6706/45279, 0, 0, 0, 1047925/1946997, -147544/196209, 1615873/1874886]
            A[8,:8] = [31/360, 0, 0, 0, 0, 64/585, 2401/7800, -1/300]
            AP = np.zeros_like(A)
            AP[1,:1] = [4/15]
            AP[2,:2] = [1/10, 3/10]
            AP[3,:3] = [3/20, 0, 9/20]
            AP[4,:4] = [9/40, 0, 0, 27/40]
            AP[5,:5] = [11/48, 0, 0, 5/8, -5/48]
            AP[6,:6] = [27112/194481, 0, 0, 56450/64827, 80000/194481, -24544/21609]
            AP[7,:7] = [-26033/41796, 0, 0, -236575/38313, -14500/10449, 275936/45279, 228095/73788]
            AP[8,:8] = [7/81, 0, 0, 0, -250/3483, 160/351, 2401/5590, 1/10]
            B = np.array([
                [31/360, 0, 0, 0, 0, 64/585, 2401/7800, -1/300, 0],
                [7/81, 0, 0, 0, -250/3483, 160/351, 2401/5590, 1/10, 0]
                ])
            E = np.array([0, 0, 0, 0, 0, 0, 0, -1/300, 1/300])
        elif method=='RKNG65':
            FSAL = False
            C = np.array([0, 1/10, 2/9, 3/7, 2/3, 4/5, 1, 1])
            A = np.zeros((8,8))
            A[1,:1] = [1/200]
            A[2,:2] = [14/2187, 40/2187]
            A[3,:3] = [148/3087, -85/3087, 1/14]
            A[4,:4] = [-2201/28350, 932/2835, -7/50, 1/9]
            A[5,:5] = [13198826/54140625, -5602364/10828125, 27987101/44687500, -332539/4021872, 1/20]
            A[6,:6] = [-601416947/162162000, 2972539/810810, 10883471/2574000, -503477/99000, 3/5, 4/5]
            A[7,:7] = [-228527046421/72442188000, 445808287/139311900, 104724572891/29896776000, -31680158501/7474194000, 1033813/2044224, 1166143/1703520, 0]
            AP = np.zeros_like(A)
            AP[1,:1] = [1/10]
            AP[2,:2] = [-2/81, 20/81]
            AP[3,:3] = [615/1372, -270/343, 1053/1372]
            AP[4,:4] = [140/297, -20/33, 42/143, 1960/3861]
            AP[5,:5] = [-15544/20625, 72/55, 1053/6875, -40768/103125, 1521/3125]
            AP[6,:6] = [6841/1584, -60/11, -15291/7436, 207319/33462, -27/8, 11125/8112]
            AP[7,:7] = [207707/54432, -305/63, -24163/14196, 4448227/821340, -4939/1680, 3837625/3066336, 0]
            B = np.array([
                [23/320, 0, 12393/54080, 2401/25350, 99/1600, 1375/32448, 0, 0],
                [23/320, 0, 111537/378560, 16807/101400, 297/1600, 6875/32448, -319/840, 9/20]
                ])
            E = np.array([
                [121541/3240000, 0, -7488783/47320000, 78566551/342225000, -102841/600000, 9349/162240, -3971/37800, 11/100],
                [17/1440, 0, -12393/189280, 40817/304200, -153/800, 2125/16224, 361/840, -9/20]
                ])
        elif method=='RKNG67':
            FSAL = True
            C = np.array([
                0,
                0.10185185185185185,
                0.1527777777777778,
                0.22916666666666666,
                0.625,
                0.375,
                0.6666666666666666,
                0.16666666666666666,
                0.9703731483217701,
                1.0,
                1.0
                ])
            A = np.zeros((11,11))
            A[1,:1] = [0.005186899862825789]
            A[2,:2] = [0.005835262345679012, 0.005835262345679012]
            A[3,:3] = [0.013129340277777778, 0.0, 0.013129340277777778]
            A[4,:4] = [0.01775568181818182, 0.0, 0.0, 0.17755681818181818]
            A[5,:5] = [0.019257489669421486, 0.0, 0.04028526829120078, 0.010349608525445846, 0.0004201335139318885]
            A[6,:6] = [0.050150891632373115, 0.0, 0.0, 0.12832080200501253, 0.014277669482347845, 0.029472859102488733]
            A[7,:7] = [0.010084019204389574, 0.0, 0.0, 0.0, -0.020329218106995884, 0.008955516362923771, 0.015178571428571428]
            A[8,:8] = [0.05878860892824075, 0.0, -0.6816741819616553, 0.0, 0.0, 0.039049081248066914, 0.13202858101777484, 0.9226199342595248]
            A[9,:9] = [0.08435639912628663, 0.0, -1.3999437419511294, 0.0, 0.0, 0.0, 0.15182217068949494, 1.6612294394931082, 0.0025357326422396132]
            A[10,:10] = [0.05359565984203665, 0.0, 0.0, 0.0, 0.0, 0.13393181352957964, 0.11449729305181633, 0.18915603352979407, 0.007915040914766067, 0.0009041591320072332]
            AP = np.zeros_like(A)
            AP[1,:1] = [0.10185185185185185]
            AP[2,:2] = [0.03819444444444445, 0.11458333333333333]
            AP[3,:3] = [0.057291666666666664, 0.0, 0.171875]
            AP[4,:4] = [0.818698347107438, 0.0, -3.137913223140496, 2.9442148760330578]
            AP[5,:5] = [0.07840909090909091, 0.0, 0.0, 0.29066985645933013, 0.0059210526315789476]
            AP[6,:6] = [0.08963711185933408, 0.0, 0.0, 0.20414673046251994, 0.1424301494476933, 0.23045267489711935]
            AP[7,:7] = [0.10180041152263375, 0.0, 0.0, 0.0, -0.3160493827160494, 0.1457965902410347, 0.23511904761904762]
            AP[8,:8] = [-1.6746873130588456, 0.0, 0.0, -7.8716193268293955, 5.3924880763160585, -1.544828210500843, -3.2555460065369957, 9.924565928931791]
            AP[9,:9] = [-3.470319856218999, 0.0, 0.0, -15.938792782884674, 11.40486380198694, -3.4698562868578473, -7.404381934689437, 19.941980772362037, -0.06349371369801968]
            AP[10,:10] = [0.05371068169618894, 0.0, 0.0, 0.0, 0.0, 0.2151684716824672, 0.34207435392349983, 0.22650729089707197, 0.3047290752184936, -0.14218987341772152]
            B = np.array([
                [0.05359565984203665, 0.0, 0.0, 0.0, 0.0, 0.13393181352957964, 0.11449729305181633, 0.18915603352979407, 0.007915040914766067, 0.0009041591320072332, 0],
                [0.05371068169618894, 0.0, 0.0, 0.0, 0.0, 0.2151684716824672, 0.34207435392349983, 0.22650729089707197, 0.3047290752184936, -0.14218987341772152, 0]
                ])
            E = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0009041591320072332, -0.0009041591320072332])
        elif method=='RKNG78':
            FSAL = True
            C = np.array([
                0,
                0.0734708048410643,
                0.1102062072615965,
                0.1653093108923948,
                0.5,
                0.2662882692912616,
                0.6337117307087383,
                0.75,
                0.5625,
                0.125,
                0.375,
                0.9665280516023570,
                1.0,
                1.0
                ])
            A = np.zeros((14,14))
            A[1,:1] = [0.0026989795819968]
            A[2,:2] = [0.0030363520297464, 0.0030363520297464]
            A[3,:3] = [0.0068317920669296, 0, 0.0068317920669296]
            A[4,:4] = [-0.0010263757731977, 0, 0, 0.1260263757731977]
            A[5,:5] = [0.0098909903843107, 0, 0.0204017587591113, 0.0050265147713328, 0.0001354572663127]
            A[6,:6] = [0.03677246469531772, 0, 0, 0.0821322947785217, 0.0300871654090989, 0.0518033539359937]
            A[7,:7] = [0.0412330490882728, 0, 0, 0.1133510029306181, 0.0567221485922376, 0.0574562020649545, 0.0124875973239167]
            A[8,:8] = [0.04214630126953125, 0, 0, 0, -0.07808807373046875, 0.1410468210292877, 0.0746038137363372, -0.0215057373046875]
            A[9,:9] = [0.0055243877171925, 0, 0, 0, 0, 0.0045913375893505, 0.0120099569922681, -0.0024361818415637, -0.0118770004572473]
            A[10,:10] = [0.0123960990923004, 0, 0, 0, 0, 0, -0.0231485688348811, 0.0044057716338592, 0.0241642368703960, 0.0524949612383254]
            A[11,:11] = [-0.1214829233717236, 0, 0, -1.5948786809469047, 0.0770898444095903, 0, 0, 0.0988449321354426, -0.1851769017765400, 1.6665727117807342, 0.5261192550365255]
            A[12,:12] = [-0.4947584676410233 , 0, 0, -5.6513209641364305, 0.4275002872904367, 0, 0, 0.3029341672695682, -1.0280329379503342, 5.4254171279669182, 1.5340242607867031, -0.0157634735858382]
            A[13,:13] = [0.0351739875863067, 0, 0, 0, 0, 0, 0, 0.0638587843542583, 0.0508667249055814, 0.1770317947276675, 0.1678171561304150, 0.0045385629257942, 0.0007129893699766]
            AP = np.zeros_like(A)
            AP[1,:1] = [0.0734708048410643]
            AP[2,:2] = [0.0275515518153991, 0.0826546554461974]
            AP[3,:3] = [0.0413273277230987, 0, 0.1239819831692961]
            AP[4,:4] = [0.8967055876379578, 0, -3.4585915268314336, 3.0618859391934758]
            AP[5,:5] = [0.0570533694239653, 0, 0, 0.2066467069549824, 0.0025881929123138]
            AP[6,:6] = [0.0221309534027327, 0, 0, 0.3666690184215938, 0.3207556053276707, -0.0758438464432589]
            AP[7,:7] = [1/12, 0, 0, 0, 0, 0.3843643696413162, 0.2823022970253504]
            AP[8,:8] = [171/2048, 0, 0, 0, 0, 0.3830674747939740, 0.1354872127060259, -81/2048]
            AP[9,:9] = [0.0734203532235939, 0, 0, 0, 0, 0.0988089649160229, 0.2415331132732774, -0.0487075617283950, -0.2400548696844993]
            AP[10,:10] = [0.0081378441127067, 0, 0, 0, 0, 0, -0.3626609117464713, 0.0697268805971279, 0.3779778062076339, 0.2818183808290027]
            AP[11,:11] = [-1.4042538922482838, 0, 0, 0, 0, -13.5555590294049575, -1.5021472824848051, 1.4767543284167949, -2.1707681965133688, 6.6149759502676558, 11.5075261735693215]
            AP[12,:12] = [-5.2708651815801315, 0, 0, 0, 0, -49.9655995536568330, -5.0302228928658231, 4.4548269045298761, -8.6071533124033841, 23.8404100463722876, 41.7115814660283881, -0.1329774764243799]
            AP[13,:13] = [0.0350993030565818, 0, 0, 0, 0, 0, 0, 0.2522347527663160, 0.1184003330687654, 0.2025813361125092, 0.2675702525942014, 0.1658638451062987, -0.0417498227046728]
            B = np.array([
                [0.0351739875863067, 0, 0, 0, 0, 0, 0, 0.0638587843542583, 0.0508667249055814, 0.1770317947276675, 0.1678171561304150, 0.0045385629257942, 0.0007129893699766, 0],
                [0.0350993030565818, 0, 0, 0, 0, 0, 0, 0.2522347527663160, 0.1184003330687654, 0.2025813361125092, 0.2675702525942014, 0.1658638451062987, -0.0417498227046728, 0]
                ])
            E = np.zeros(14)
            E[[-2,-1]] = [0.0007129893699766, -0.0007129893699766]
        else:
            pass
        return C, A, AP, B, E, FSAL
    
    
    def _rkng_step(self, fun, t, y, yp, f, h, rtol, atol, C, A, AP, B, E, FSAL, method):
        states = len(y)
        stages = A.shape[0]
        F = np.zeros((
            stages, states
            ))
        t_new = t + h
        
        if FSAL:
            stages -= 1
        
        F[0] = f
        if method in ['RKNG5456', 'RKNG6459', 'RKNG5459']:
            # Exploit the special structure of the RKNG tableau having
            # fewer y stages than y' stages
            if method=='RKNG5456':
                for i in range(1,5):
                    F[i] = fun(t + C[i]*h,
                               y + C[i]*h*yp + h**2 * A[i,:i] @ F[:i],
                               yp + h * AP[i,:i] @ F[:i]
                               )
                y_new = y + h*yp + h**2 * B[0,:5] @ F[:5]
                F[5] = fun(t + C[5]*h,
                           y_new,
                           yp + h * AP[5,:5] @ F[:5]
                           )
                yp_new = yp + h * B[1,:6] @ F[:6]
                f_new = fun(t_new, y_new, yp_new)
                F[6] = f_new
            # Both RKNG6459 and RKNG5459 have the same evaluation patterns
            else:
                Y = y + C[1]*h*yp + h**2 * A[1,:1] @ F[:1]
                for i in range(1,3):
                    F[i] = fun(t + C[i]*h,
                               Y,
                               yp + h * AP[i,:i] @ F[:i]
                               )
                F[3] = fun(t + C[3]*h,
                           y + C[3]*h*yp + h**2 * A[3,:3] @ F[:3],
                           yp + h * AP[3,:3] @ F[:3]
                           )
                Y = y + C[4]*h*yp + h**2 * A[4,:4] @ F[:4]
                for i in range(4,6):
                    F[i] = fun(t + C[i]*h,
                               Y,
                               yp + h * AP[i,:i] @ F[:i]
                               )
                Y = y + C[6]*h*yp + h**2 * A[6,:6] @ F[:6]
                for i in range(6,8):
                    F[i] = fun(t + C[i]*h,
                               Y,
                               yp + h * AP[i,:i] @ F[:i]
                               )
                y_new = y + h*yp + h**2 * B[0,:8] @ F[:8]
                F[8] = fun(t + C[8]*h, y_new, yp + h * AP[8,:8] @ F[:8])
                yp_new = yp + h * B[1,:9] @ F[:9]
                f_new = fun(t_new, y_new, yp_new)
                F[9] = f_new
        else:
            for i in range(1,stages):
                F[i] = fun(
                    t + C[i]*h,
                    y + C[i]*h*yp + h**2 * A[i,:i] @ F[:i],
                    yp + h * AP[i,:i] @ F[:i]
                    )
            if FSAL:
                y_new = y + h*yp + h**2 * B[0,:-1] @ F[:-1]
                yp_new = yp + h * B[1,:-1] @ F[:-1]
                f_new = fun(t_new, y_new, yp_new)
                F[-1] = f_new
            else:
                y_new = y + h*yp + h**2 * B[0] @ F
                yp_new = yp + h * B[1] @ F
                f_new = fun(t_new, y_new, yp_new)
        # Estimate error
        sc = np.zeros((2,states))
        sc[0] = atol + rtol * np.maximum(
            np.abs(y),
            np.abs(y_new)
            )
        sc[1] = atol + rtol * np.maximum(
            np.abs(yp),
            np.abs(yp_new)
            )
        if E.ndim==1:
            err = E @ F * h**2
            error_norm = RMS(err/sc[0],0)
        else:
            err = np.zeros((2,states))
            err[0] = E[0] @ F * h**2
            err[1] = E[1] @ F * h
            error_norm = RMS(err/sc,1).max()
        return t_new, y_new, yp_new, f_new, error_norm
    
    
    def _integrate(self, fun, tspan, y0, yp0, method, h, hmax, hmin, rtol, atol, maxiter):
        MAX_FACTOR = 10.
        MIN_FACTOR = .1
        SAFETY = .9
        p, order = {'RKNG4':(3,4),
                    'RKNG43':(3,4),
                    'RKNG54':(4,5),
                    'RKNG5456':(4,5),
                    'RKNG5459':(4,5),
                    'RKNG6459':(4,6),
                    'RKNG56':(5,5),
                    'RKNG65':(5,6),
                    'RKNG67':(6,6),
                    'RKNG78':(7,7)
                    }[method]
        EXPONENT = -1/(p+1)
        KI = -.7/order
        KP = .4/order
        
        t0, tf = tspan
        f0 = fun(t0,y0,yp0)
        
        if h is None:
            h = self._compute_first_step(fun, t0, y0, yp0, f0, p, rtol, atol)
        
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
                t_new, y_new, yp_new, f_new, _ = self._rkng_step(fun, t_old, y_old, yp_old, f_old, h, rtol, atol, self.C, self.A, self.AP, self.B, self.E, self.FSAL, method)
                t = np.hstack(( t, t_new ))
                y = np.vstack(( y, y_new ))
                yp = np.vstack(( yp, yp_new ))
                f = np.vstack(( f, f_new ))
                hacc += 1
                break
            
            t_new, y_new, yp_new, f_new, error_norm = self._rkng_step(fun, t_old, y_old, yp_old, f_old, h, rtol, atol, self.C, self.A, self.AP, self.B, self.E, self.FSAL, method)
            
            if error_norm < 1 or h==hmin:
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
    
    
    def _spline_interpolant(self, fun, t, y, yp, f, rtol, atol, method, interpolant):
        if interpolant==5:
            Y = np.zeros((t.size, 3, y.shape[1]))
            Y[:,0,:] = y
            Y[:,1,:] = yp
            Y[:,2,:] = f
        else:
            steps = t.size - 1
            h = np.diff(t)
            k = y.shape[1]
            y_mid = np.zeros((steps, k))
            yp_mid = np.zeros_like(y_mid)
            f_mid = np.zeros_like(y_mid)
            d3y = np.zeros_like(f)
            d4y = np.zeros_like(f)
            for i in range(steps):
                _, y_mid[i], yp_mid[i], f_mid[i], _ = self._rkng_step(fun, t[i], y[i], yp[i], f[i], .5*h[i], rtol, atol, self.C, self.A, self.AP, self.B, self.E, self.FSAL, method)
                d3y[i] = 3*(
                    h[i]**2*(-9*f[i] + 16*f_mid[i] - f[i+1])
                    +
                    h[i]*(-96*yp[i] - 64*yp_mid[i] + 20*yp[i+1])
                    +
                    (-396*y[i] + 512*y_mid[i] - 116*y[i+1])
                    ) / h[i]**3
                d4y[i] = 12*(
                    h[i]**2*(33*f[i] - 112*f_mid[i] + 8*f[i+1])
                    +
                    h[i]*(468*yp[i] + 320*yp_mid[i] - 158*yp[i+1])
                    +
                    (2166*y[i] - 3072*y_mid[i] + 906*y[i+1])
                    ) / h[i]**4
            d3y[-1] = 3*(
                h[-1]**2*(f[-2] - 16*f_mid[-1] + 9*f[-1])
                +
                h[-1]*(20*yp[-2] - 64*yp_mid[-1] - 96*yp[-1])
                +
                (116*y[-2] - 512*y_mid[-1] + 396*y[-1])
                ) / h[-1]**3
            d4y[-1] = 12*(
                h[-1]**2*(8*f[-2] - 112*f_mid[-1] + 33*f[-1])
                +
                h[i]*(158*yp[-2] - 320*yp_mid[-1] - 468*yp[-1])
                +
                (906*y[-2] - 3072*y_mid[-1] + 2166*y[-1])
                ) / h[i]**4
            Y = np.zeros((t.size, 5, k))
            Y[:,0,:] = y
            Y[:,1,:] = yp
            Y[:,2,:] = f
            Y[:,3,:] = d3y
            Y[:,4,:] = d4y
        return BPoly.from_derivatives(t, Y)
    
    
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
    
    
    @staticmethod
    def _validate_misc(maxiter,method,interpolant):
        if type(maxiter)!=int:
            raise TypeError('maxiter argument must be an integer.')
            if maxiter < 1:
                raise ValueError('maxiter argument must be greater than zero.')
        method_list = ['RKNG4', 'RKNG43','RKNG54', 'RKNG5456', 'RKNG5459',
                       'RKNG6459', 'RKNG56', 'RKNG65','RKNG67','RKNG78']
        if method not in method_list:
            raise Exception(f"Value of method not in {method_list}")
        if interpolant not in [5,8]:
            raise Exception(f"Value of interpolant not in {[5,8]}")
        return None
