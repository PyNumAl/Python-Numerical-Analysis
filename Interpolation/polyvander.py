import numpy as np
from scipy.special import factorial
from numpy.linalg import solve

def polyvander(xp,yp):
    """
    Polynomial Interpolator
    
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
    UP, P = np.meshgrid(up, p, indexing='ij')
    V = UP ** P
    coeff = solve(V,yp)
    
    def P(x,k=0):
        if coeff.ndim==1:
            return horner(du_dx*(x-a)-1, coeff, k) * du_dx ** k
        else:
            return np.array([ horner(du_dx*(x-a)-1, coeff[:,i], k) * du_dx ** k for i in range(coeff.shape[1])]).T
    return P

def horner(x, a, k=0):
    """
    Recursive Horner's rule
    
    Evaluate efficiently the Vandermonde polynomial or its derivatives/antiderivatives.
    
    Parameters
    --------
    x : int, float, or complex
        Point of evaluation
        
    a : array_like, shape(n,)
        Vandermonde coefficients
        
    k : int
        The derivative/antiderivative order
        
    Returns
    --------
    val : int, float, or complex
        The value of the k-th derivative/antiderivative of the polynomial given by the coefficients a at x.
    """
    a = np.asarray(a)
    if a.ndim==0:
        return horner(x, [a.item()], k)
    if a.ndim>1:
        return np.array([horner(x, a[i], k) for i in range(a.shape[0])])
    else:
        if isinstance(x, list) or isinstance(x,tuple):
            x = np.asarray(x)/1.0
        
        if type(x)==int:
            x /= 1.0
        
        if type(k) is not int:
            raise TypeError('Argument k must be an integer.')
        else:
            pass
        
        n = len(a)
        d = n - 1
        if k > d:
            val = np.zeros_like(x)
        elif k == d:
            val = np.full_like(x, factorial(k) * a[k])
        elif k == 0:
            if n==1:
                val = np.full_like(x, a[0])
            elif n==2:
                val = a[0] + x * a[1]
            else:
                val = a[0] + x * horner(x, a[1:], k)
        elif k<d and k>0:
            val = horner(x, a[1:] * np.arange(1,n), k-1)
        elif k<0:
            val = horner(x, np.hstack(( 0, a / np.arange(1,n+1) )), k+1)
        else:
            pass
        return val
