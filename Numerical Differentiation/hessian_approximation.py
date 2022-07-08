import numpy as np

EPS = np.finfo(float).eps

def nth_root(x,n):
    return x ** (1/n)

def generate_step(x, order, deriv_order):
    hdefault = nth_root(EPS, order + deriv_order)
    step= hdefault*(np.abs(x) + 1)
    return step

def fhess(f, x, h=None, method=2, args=None):
    """
    Approximate the hessian of a scalar multivariable function using real or complex-step finite differences.
    
    Complex-step differentiation is the most accurate and efficient method provided that the function f is analytic.
    The complex-step finite difference method (CSFDM) used is Eq.(9) in [3], requiring only two function evaluations:
        
        fprime2(x) = Im( f(x + h + iv) - f(x - h + iv) ) / 2 / h / v
        
    From the equation above, the complex step, h + iv, cannot be purely an imaginary step. Otherwise, division by zero
    occurs. If the real and imaginary parts, h and v, are equal, then CSFDM is 4th-order
    accurate using only two (2) function evaluations per element of the hessian. Otherwise, CSFDM is only 2nd-order
    accurate for h!=v. In either case, because there is a difference involved, the complex-step cannot be made
    arbitrarily small due to subtractive cancellation.￼ A fairly good estimate for the optimal complex step is:
        
        FAC = (EPS) ** (1/6) = 0.00246
        step = FAC * (1 + 1j) or step = 1e-3*(1 + 1j)
    
    
    Parameters
    --------
    f : callable
        Scalar multivariable function whose hessian is to be approximated.
        
    x : array_like
        Real-valued vector on which to approximate the hessian.
        
    h : int, float, or complex scalar or array_like, optional
        The stepsize to use for gradient approximation. Simply pass in complex numbers to use
        complex-step differentiation.
    
    method : {1, 2, 4}, optional
        The finite difference method to use for approximation of the hessian. 
        Methods 1, 2, and 4 correspond to 1st, 2nd, and 4th-order finite difference methods.
        If not given, the default value is to use 2nd-order finite differences. If a complex number/s
        were given for h, complex-step finite difference method is used and the value for order is ignored.
        
    args : list or tuple, optional
        Additional parameters to pass to the function. Cannot be an empty list or tuple.
    
    
    Returns
    --------
    H : 2-d array
        The Hessian approximation.
    
    
    References
    --------
    [1] Scilab. numderivative. https://help.scilab.org/docs/6.1.1/en_US/numderivative.html
    [2] Scilab. (2017). https://github.com/opencollab/scilab/blob/master/scilab/modules/differential_equations/macros/numderivative.sci
    [3] Rafael Abreu, Daniel Stich, Jose Morales, The Complex-Step-Finite-Difference method, Geophysical Journal International, Volume 202, Issue 1, July 2015, Pages 72–93, https://doi.org/10.1093/gji/ggv125
    [4] Numdifftools. (2019). 5.1.1.5. numdifftools.core.Hessian. https://numdifftools.readthedocs.io/en/latest/reference/generated/numdifftools.core.Hessian.html
    """
    f, fx, H = prepare_input_fhess(f, x, h, method, args)
    n = len(H)
    Hessian = np.zeros((n,n))
    for i in range(n):
        Hi = H[i]
        Hii = H[i,i]
        for j in range(i,n):
            Hj = H[j]
            Hjj = H[j,j]
            if H.dtype==complex:
                val = np.matmul([1, -1], [f(x + Hi.real + Hj.imag*1j), f(x - Hi.real + Hj.imag*1j)]
                                ).imag / 2 / Hii.real / Hjj.imag
            else:
                if method==1:
                    diff1 = np.matmul([1, -1], [f(x + Hj + Hi), f(x + Hj)])
                    diff2 = np.matmul([1, -1], [f(x + Hi), fx])
                    val = ( diff1 - diff2 ) / Hii / Hjj
                elif method==2:
                    diff1 = np.matmul([1, -1], [f(x + Hj + Hi), f(x + Hj - Hi)])
                    diff2 = np.matmul([1, -1], [f(x - Hj + Hi), f(x - Hj - Hi)])
                    val = ( diff1 - diff2 ) / 4 / Hii / Hjj
                else:
                    diff1 = np.matmul([1, -8, 8, -1], [f(x - 2*Hj - 2*Hi),
                                                       f(x - 2*Hj - Hi),
                                                       f(x - 2*Hj + Hi),
                                                       f(x - 2*Hj + 2*Hi)
                                                       ]
                                      )
                    diff2 = np.matmul([1, -8, 8, -1], [f(x - Hj - 2*Hi),
                                                       f(x - Hj - Hi),
                                                       f(x - Hj + Hi),
                                                       f(x - Hj + 2*Hi)
                                                       ]
                                      )
                    diff3 = np.matmul([1, -8, 8, -1], [f(x + Hj - 2*Hi),
                                                       f(x + Hj - Hi),
                                                       f(x + Hj + Hi),
                                                       f(x + Hj + 2*Hi)
                                                       ]
                                      )
                    diff4 = np.matmul([1, -8, 8, -1], [f(x + 2*Hj - 2*Hi),
                                                       f(x + 2*Hj - Hi),
                                                       f(x + 2*Hj + Hi),
                                                       f(x + 2*Hj + 2*Hi)
                                                       ]
                                      )
                    val = ( diff1 - 8*diff2 + 8*diff3 - diff4 ) / 144 / Hii / Hjj
            Hessian[i,j] = val
            if i!=j:
                Hessian[j,i] = val
            else:
                pass
    return Hessian

# Vectorized hessian function
vfhess = np.vectorize(fhess, signature='(n)->(n,n)', excluded=[0, 2, 3, 4])

def prepare_input_fhess(f, x, h, method, args):
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
        h = generate_step(x, method, 2)
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