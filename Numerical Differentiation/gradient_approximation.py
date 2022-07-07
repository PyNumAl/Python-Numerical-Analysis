# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 15:16:37 2022

@author: raqui
"""

import numpy as np

EPS = np.finfo(float).eps

def nth_root(x,n):
    return x ** (1/n)

def generate_step(x, order, deriv_order):
    hdefault = nth_root(EPS, order + deriv_order)
    step= hdefault*(np.abs(x) + 1)
    return step

def fgrad(f, x, h=None, method=2, args=None):
    """
    Approximate the gradient of a scalar multivariable function using real or complex-step finite differences.
    
    Complex-step differentiation is the most accurate and efficient method provided that the function f is analytic.
    The complex-step finite difference method (CSFDM) used is Eq.(5) in [3], requiring only one function evaluation:
        
        fprime(x) = Im(f(x + h + vi)) / v
        
    If the provided complex step, h = a + bj, has a non-zero real component, then the CSFDM approximation is
    only 1st-order. Otherwise, it is 2nd-order. In either case, the complex-step, the imaginary component v in
    particular, can be made almost arbitrarily small as possible without subtractive cancellation occuring,
    allowing one to obtain exact approximations.
    
    In my own numerical tests, for a purely imaginary step, h = bj, a value of 1e-324 for b yields not-a-number (nan)
    results while loss of significant digits occurs at the immediate powers before 1e-324 i.e.
    1e-323, 1e-322, 1e-321, 1e-320, etc. Setting b to be as small as the machine precision EPS, approximately
    2.22e-16, is more than enough to achieve exact results.
    
    Parameters
    --------
    f : callable
        Scalar multivariable function whose gradient is to be approximated.
        
    x : array_like
        Real-valued vector on which to approximate the gradient.
        
    h : int, float, or complex scalar or array_like, optional
        The stepsize to use for gradient approximation. Simply pass in complex numbers to use
        complex-step differentiation.
    
    method : {1, 2, 4}, optional
        The finite difference method to use for approximation of the gradient. 
        Methods 1, 2, and 4 correspond to 1st, 2nd, and 4th-order finite difference methods.
        If not given, the default value is to use 2nd-order finite differences. If a complex number/s
        were given for h, complex-step finite difference method is used and the value for order is ignored.
        
    args : list or tuple, optional
        Additional parameters to pass to the function. Cannot be an empty list or tuple.
    
    
    Returns
    --------
    G : 1-d array
        The Gradient approximation.
    
    
    References
    --------
    [1] numderivative. https://help.scilab.org/docs/6.1.1/en_US/numderivative.html
    [2] Scilab. (2017). https://github.com/opencollab/scilab/blob/master/scilab/modules/differential_equations/macros/numderivative.sci
    [3] Rafael Abreu, Daniel Stich, Jose Morales, The Complex-Step-Finite-Difference method, Geophysical Journal International, Volume 202, Issue 1, July 2015, Pages 72–93, https://doi.org/10.1093/gji/ggv125
    """
    f, fx, H = prepare_input_fgrad(f, x, h, method, args)
    n = len(H)
    Grad = np.zeros(n)
    for i in range(n):
        if H.dtype == complex:
            Grad[i] = np.matmul([1],
                               [f(x + H[i])]
                               ).imag / H[i,i].imag
        else:
            if method==1:
                Grad[i] = np.matmul([-1, 1],
                                   [fx, f(x + H[i])]
                                   ) / H[i,i]
            elif method==2:
                Grad[i] = np.matmul([-1, 0, 1],
                                   [f(x - H[i]), fx, f(x + H[i])]
                                   ) / (2*H[i,i])
            else:
                Grad[i] = np.matmul([1, -8, 0, 8, -1],
                                   [f(x - 2*H[i]), f(x - H[i]), fx,
                                    f(x + H[i]), f(x + 2*H[i])]
                                   ) / (12*H[i,i])
    return Grad

# Vectorized gradient function
vfgrad = np.vectorize(fgrad, signature='(n)->(n)', excluded=[0, 2, 3, 4])

def prepare_input_fgrad(f, x, h, method, args):
    if args is not None:
        if isinstance(args, tuple) or isinstance(args, list) and len(args)>0:
            f = lambda x, f=f: f(x, *args)
        else:
            raise Exception('args must be a list or tuple of at least length 1 containing the additional parameters to be passed to the function.')
    else:
        pass
    
    if method not in [1, 2, 4]:
        raise Exception(f'Invalid value for method. Value of order must be among {[1, 2, 4]}.')
    else:
        pass
    
    x = np.atleast_1d(x)
    if x.ndim!=1:
        raise ValueError('x must be a 1d vector of values.')
    elif x.dtype in [int, float, complex]:
        if x.dtype==int or x.dtype==float:
            pass
        else:
            raise ValueError('Complex values not allowed for x. x must be a real-valued vector.')
    else:
        raise TypeError('x must be a real-valued vector.')
    
    if h is None:
        h = generate_step(x, method, 1)
    else:
        h = np.asarray(h)
        if h.ndim!=0 and h.ndim!=1:
            raise ValueError(f'If h is not {None}, it must be a scalar or vector of shape {x.shape}.')
        elif h.ndim==1 and h.shape!=x.shape:
            raise ValueError(f'Incompatible shapes for h and x, {h.shape} and {x.shape}.')
        elif h.dtype not in [int, float, complex]:
            raise TypeError('Invalid input type for h. h must be a scalar or vector of int, float, or complex numbers.')
        elif h.ndim==0:
            h = h * np.ones_like(x)
        else:
            pass
    
    H = np.diag(h)
    
    fx = np.asarray(f(x))
    if fx.ndim!=0:
        raise ValueError('Output of f must be a scalar.')
    elif fx.dtype not in [int, float]:
        raise TypeError('Invalid output type for f. f is expected to be a real-valued scalar multivariable function for real-valued vector inputs.')
    else:
        pass
    return f, fx, H