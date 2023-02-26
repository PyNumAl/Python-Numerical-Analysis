import numpy as np
from numpy.linalg import solve
from numpy.polynomial import polynomial as P

def vanderpoly(xp,yp):
    """
    Vandermonde Polynomial Interpolator
    
    Interpolate sets of datapoints using polynomials in the Vandermonde basis. The resulting
    polynomial and its derivatives and antiderivatives is efficiently evaluated using
    recursive horner's rule.
    
    Parameters
    --------
    xp : array_like, shape(n,)
        1-D array containing values of the independent variable. Values don't have to be
        in strictly increasing order since they are sorted internally along with the
        corresponding values of the dependent variable. However, the values must be real
        and finite for there to be sensible results.
        
        For numerical stability, the values of the independent variable are linearly mapped to the
        interval [-1,1]. The inputs to the resulting polynomial output function are also similarly
        transformed. The constant 2/(b-a), raised to the power of k, is multiplied to the output
        to obtain the correct value of interpolator in the original xp domain. a and b are the
        left and right endpoints of the interval, respectively, while k is the order of the
        desired derivative/antiderivative.
        
    yp: array_like, shape(n,) or shape(n,m)
        1-D or 2-D array containing values of the independent variable. Values must be real
        and finite. For complex values, simply seperate into real and imaginary parts and
        then interpolate as usual.
        
    
    Returns
    --------
    P : callable
        Function evaluating the polynomial interpolator and its derivatives and anti-derivatives.
    """
    xp = np.asfarray(xp)
    yp = np.asfarray(yp)
    idx = np.argsort(xp)
    xp = xp[idx]
    yp = yp[idx]
    
    a, b = xp[[0,-1]]
    du_dx = 2/(b-a)
    up = du_dx*(xp-a)-1
    
    p = np.arange(up.size).astype(float)
    UP, PP = np.meshgrid(up, p, indexing='ij')
    V = UP ** PP
    coeff = solve(V,yp)
    
    def Pfun(x,m=0, k=[], lbnd=0):
        u = du_dx*(x-a)-1
        if m==0:
            val = P.polyval(u, coeff)
        elif m>0:
            val = P.polyval(u, P.polyder(coeff, m, du_dx))
        else:
            val = P.polyval(u, P.polyint(coeff, abs(m), k, lbnd, 1/du_dx))
        return val
    return Pfun
