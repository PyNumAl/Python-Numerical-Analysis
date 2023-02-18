import sympy as sp
import numpy as np
from numpy.linalg import solve
import mpmath as mpm
from mpmath import mp, mpf, mpc, nprint, chop
from sympy import Rational as R
from scipy.special import factorial

eps = np.finfo(float).eps

def fdcc(stencil, d=None, symbolic=True, dps=16):
    """
    Finite Difference Coefficients Calculator for computing finite difference coefficients
    symbolically or through double and arbitrary precision.
    
    Note: This function only handles real-valued stencils for now. In the future, either it will
    be extended to handle complex stencils for computing complex-step finite difference coefficients;
    or another function will be made to handle specifically complex stencils.
    
    Parameters
    --------
    stencil: array-like
        The finite difference stencil. The stencil doesn't have to be increasing or decreasing,
        however, the each stencil point must be unique or else the resulting linear system
        is unsolvable due to a singular matrix.
        
    d: int or array-like, optional
        The desired derivative order to approximate. If none, then all possible derivatives are estimated.
        A value of zero corresponds to interpolation.
        
    symbolic: bool, optional
        Whether to compute the coefficients symbolically or numerically.
        
    dps: int, optional
        The desired number of decimal points in the coefficients. This argument is ignored
        when symbolic is True. mpmath functions are used for dps > 16. Numpy is used otherwise
        for fast array computations.
        
    Returns
    --------
    coeff: SymPy Matrix, mpmath matrix, or numpy array
        In the general case, it is a 2-d matrix/array whose columns contain the finite difference
        coefficients, with the j-th column corresponding to the j-th argument of the input d.
        
        If symbolic is False, dps is 16, and only one derivative approximation is desired i.e. input d has only one element,
        then the output is a 1-d numpy array.
        
    series_terms: SymPy Matrix, mpmath matrix, or numpy array
        2-d matrix/array of taylor series expansions, with the j-th column corresponding to the
        taylor series expansion of the j-th column in output coeff.
        
        If symbolic is False, dps is 16, and only one derivative approximation is desired i.e. input d has only one element,
        then the output is a 1-d numpy array.
        
        The number of terms in the taylor
        series expansions is the length of the finite difference stencil plus two (2). This allows
        one to deduce the order of the finite difference approximations.
    
    Examples
    --------
    >>> coeff, series_terms = fdcc([2,1,0,-1,-2])
    >>> coeff
    Matrix([
    [0, -1/12, -1/12,  1/2,  1],
    [0,   2/3,   4/3,   -1, -4],
    [1,     0,  -5/2,    0,  6],
    [0,  -2/3,   4/3,    1, -4],
    [0,  1/12, -1/12, -1/2,  1]])
    >>> series_terms
    Matrix([
    [1,     0,     0,   0,   0],
    [0,     1,     0,   0,   0],
    [0,     0,     1,   0,   0],
    [0,     0,     0,   1,   0],
    [0,     0,     0,   0,   1],
    [0, -1/30,     0, 1/4,   0],
    [0,     0, -1/90,   0, 1/6]])
    
    >>> coeff, series_terms = fdcc([2,1,0,-1,-2], 1, False) # coefficients for the 1st derivative only
    >>> coeff
    array([-0.08333333,  0.66666667,  0.        , -0.66666667,  0.08333333])
    >>> series_terms
    array([ 0.        ,  1.        ,  0.        ,  0.        ,  0.        ,
           -0.03333333,  0.        ])
    
    >>> coeff, series_terms = fdcc([2,1,0,-1,-2], [3,4], False) # coefficients for both 3rd and 4th derivatives
    >>> coeff
    array([[ 0.5,  1. ],
           [-1. , -4. ],
           [ 0. ,  6. ],
           [ 1. , -4. ],
           [-0.5,  1. ]])
    >>> series_terms
    array([[0.        , 0.        ],
           [0.        , 0.        ],
           [0.        , 0.        ],
           [1.        , 0.        ],
           [0.        , 1.        ],
           [0.25      , 0.        ],
           [0.        , 0.16666667]])
    
    Use the function cheb_nodes to generate a stencil based on chebyshev nodes
    to arbitrary precision.
    
    >>> stencil = cheb_nodes(a=-2, b=2, points=5, symbolic=False, dps=25)
    >>> nprint(stencil, 25)
    [2.0, 1.732050807568877293527446, 0.0, -1.732050807568877293527446, -2.0]
    >>> coeff, series_terms = fdcc(stencil, [1,2,3,4], False, 25) # solve the coefficients with the same dps of 25
    >>> nprint(coeff, 16)
    [             -0.75               -0.75                 1.5   3.0]
    [ 1.154700538379252   1.333333333333333  -1.732050807568877  -4.0]
    [               0.0  -1.166666666666667                 0.0   2.0]
    [-1.154700538379252   1.333333333333333   1.732050807568877  -4.0]
    [              0.75               -0.75                -1.5   3.0]
    >>> nprint(series_terms, 8)
    [ 0.0           0.0   0.0         0.0]
    [ 1.0           0.0   0.0         0.0]
    [ 0.0           1.0   0.0         0.0]
    [ 0.0           0.0   1.0         0.0]
    [ 0.0           0.0   0.0         1.0]
    [-0.1           0.0  0.35         0.0]
    [ 0.0  -0.033333333   0.0  0.23333333]
    """
    stencil = np.asarray(stencil)
    s = stencil.size
    if stencil.dtype not in [int, float, np.int32, np.int64, np.float32, np.float64, mpm.ctx_mp_python.mpf]:
        raise TypeError('Invalid type for stencil argument.')
    elif stencil.ndim!=1:
        raise Exception('Argument stencil must be 1-d array-like.')
    elif not s>1:
        raise Exception('Stencil length must be greater than 1.')
    else:
        pass
    
    if d is not None:
        d = np.atleast_1d(d)
        if d.ndim!=1:
            raise Exception('Argument d must be 1-d array-like.')
        if d.dtype!=int:
            raise TypeError('Argument d must be an integer/array of integers')
        elif np.any(d<0):
            raise ValueError('Values for d must be non-negative integers.')
        elif np.any(d>=s):
            idx = d>=s
            raise Exception(f'Derivative orders {d[idx]} does not exist for stencil of length {s}.')
        else:
            pass
    else:
        pass
    
    if type(dps) is not int:
        raise TypeError('Argument dps must be an integer')
    elif dps<16:
        raise ValueError('Value of dps must be at least 16.')
    else:
        pass
    
    if type(symbolic) is not bool:
        raise TypeError
    else:
        pass
    
    points = s
    terms = points + 2
    if symbolic:
        if d is None:
            rhs = sp.eye(points)
        elif d.size==1:
            d = d.item()
            rhs = sp.Matrix([1 if i==d else 0 for i in range(points)])
        else:
            rhs = sp.Matrix([[1 if i==di else 0 for i in range(points)] for di in d]).T
        A = sp.Matrix([
            [
                stencil[j]**i / sp.factorial(i) for j in range(points)
                ] for i in range(terms)
            ])
        coeff = A[:points,:]**-1 * rhs
        series_terms = A * coeff
        
        coeff = sp.simplify(coeff)
        series_terms = sp.simplify(series_terms)
    elif dps==16:
        if d is None:
            rhs = np.eye(points)
        elif d.size==1:
            d = d.item()
            rhs = np.array([1 if i==d else 0 for i in range(points)])
        else:
            rhs = np.array([[1 if i==di else 0 for i in range(points)]for di in d]).T
        powers = np.arange(terms, dtype=float)
        POWERS, STENCIL = np.meshgrid(powers, stencil, indexing='ij')
        A = STENCIL ** POWERS
        coeff = solve(
            A[:points,:],
            np.diag(factorial(powers[:points])) @ rhs
            )
        series_terms = np.diag(1/factorial(powers)) @ A @ coeff
        
        coeff = np.where(np.abs(coeff)<=1e3*eps, 0, coeff)
        series_terms = np.where(np.abs(series_terms)<=1e3*eps, 0, series_terms)
        
    else:
        mp.dps = dps
        if d is None:
            rhs = mpm.eye(points)
        elif d.size==1:
            d = d.item()
            rhs = mpm.matrix([1 if i==d else 0 for i in range(points)])
        else:
            rhs = mpm.matrix([[1 if i==di else 0 for i in range(points)] for di in d]).T
        A = mpm.matrix([
            [
                mpm.power(stencil[j], i) / mpm.factorial(i) for j in range(points)
                ] for i in range(terms)
            ])
        coeff = A[:points,:]**-1 * rhs
        series_terms = A * coeff
        
        coeff = mpm.chop(coeff, mpm.power(10,3-dps))
        series_terms = mpm.chop(series_terms, mpm.power(10,3-dps))
    
    return coeff, series_terms

def cheb_nodes(a=-1, b=1, points=3, symbolic=True, dps=16):
    """
    Function for generating stencils based on Chebyshev nodes
    """
    if type(points) is not int:
        raise TypeError
    elif points<2:
        raise ValueError
    elif points==2:
        return [a,b]
    else:
        pass
    n = points - 2
    
    if type(dps) is not int:
        raise TypeError
    elif dps<16:
        raise ValueError
    else:
        pass
    
    if type(symbolic) is not bool:
        raise TypeError
    elif symbolic is True:
        out = [b] + [
            R(a+b,2) + R(b-a,2)*sp.cos( (2*k-1)*sp.pi/2/n ) for k in range(1,n+1)
            ] + [a]
    else:
        mp.dps = dps
        out = [b] + [mpm.fraction(a+b,2) + mpm.fraction(b-a,2)*mpm.cos( mpm.fraction(2*k-1, 2*n)*mpm.pi ) for k in mpm.arange(1,n+1)] + [a]
        out = mpm.chop(out)
        if dps==16:
            out = np.asfarray(out)
    return out
