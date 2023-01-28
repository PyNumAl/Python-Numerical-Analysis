import numpy as np
from numba import njit

eps = np.finfo(float).eps

def validate_inputs(f, x, h, order, H_form, zero_tol, args):
    if args is None:
        f = lambda u, f=f: np.asarray( f(u) )
    else:
        args_type = (isinstance(args,list) or  isinstance(args,tuple))
        if not args_type:
            raise TypeError('Argument args must be a tuple or list.')
        elif args_type and len(args)==0:
            raise ValueError('Argument args must be a non-empty tuple or list.')
        else:
            f = lambda u, f=f: np.asarray( f(u,*args) )
    
    if zero_tol is not None:
        if type(zero_tol) is not float:
            raise TypeError('Argument zero_tol must be a float.')
        elif zero_tol<=0 or zero_tol>=1:
            raise ValueError('Argument zero_tol must be positive number smaller than one.')
        else:
            pass
    else:
        zero_tol = max(eps ** (order/(order+2)), 1e3*eps)
    
    if type(H_form) is not str:
        raise TypeError('Argument H_form must be a string.')
    elif H_form not in ['default', 'blockmat', 'hypermat']:
        raise ValueError(f"Value of H_form must be among {['default', 'blockmat', 'hypermat']}")
    else:
        pass
    
    x = np.asarray(x)
    if x.ndim!=1:
        raise Exception('Input x must be 1-d array-like.')
    elif x.dtype==complex:
        raise TypeError('Complex values not allowed. x must be a vector of real numbers.')
    elif x.dtype!=int and x.dtype!=float:
        raise TypeError('x values must be integers or floats.')
    else:
        pass
    
    fx = f(x)
    if fx.ndim>1:
        raise Exception('Output of f must be a scalar or 1-d.')
    elif fx.dtype==complex:
        raise TypeError('Complex values not allowed. Output of f must be a vector of real numbers.')
    elif fx.dtype!=int and fx.dtype!=float:
        raise TypeError('Output of f must be real numbers of type int or float.')
    else:
        pass
    
    n = x.size
    m = {0:fx.ndim, 1:fx.size}[fx.ndim]
    
    if h is None:
        h_def = eps ** (1/(order + 2))
        h = h_def*(1 + np.abs(x))
    else:
        h = np.asarray(h)
        if h.ndim!=0 and h.ndim!=1:
            raise Exception('Argument h must be a scalar or a vector.')
        elif h.dtype!=int and h.dtype!=float:
            raise TypeError('Argument h must be a scalar or vector of int or float.')
        elif h.ndim==1 and h.size!=n:
            raise Exception('h must have the same length as x if it is a vector')
        elif h.ndim==0:
            h = np.full(n, h)
        else:
            pass
    h = np.diag(h)
    
    if type(order) is not int:
        raise TypeError('Argument order must be an integer.')
    elif order % 2 != 0:
        raise ValueError('Argument order must be an even integer.')
    elif order not in [2, 4, 6, 8, 10]:
        raise Exception(f"Value for order must be in {[2, 4, 6, 8, 10]}")
    else:
        pass
    
    return f, x, h, order, H_form, zero_tol, fx, n, m

@njit(['f8[:,:](f8[:,:,:])'], cache=True, boundscheck=True)
def blockmat_reshape(H):
    m = H.shape[0]
    n = H.shape[1]
    H_new = np.zeros((m*n,n))
    lb = 0
    ub = n
    for i in range(m):
        H_new[lb:ub,:] = H[i]
        lb += n
        ub += n
    return H_new

def numderivative(f, x, h=None, order=2, H_form='default', zero_tol=None, args=None):
    """
    Simultaneously approximate the jacobian and hessian of a real-valued vector function using central finite differences.
    
    Python version of SciLab's numderivative function. What makes this function unique is the ability
    to approximate the hessian of a vector function of vector inputs. For a scalar function of vector inputs,
    the hessian is a 2-d array or matrix. The hessian of a vector function is a 3-d array and it can be
    considered as a vector of hessian matrices.
    
    No function exists in SciPy for approximating 3-d hessians. The Hessian function of numdifftools [3]
    only approximates the hessian of a scalar multi-variable function. Thus, this function fills a niche
    that hasn't been met. A specific, important application for 3-d hessian matrices is in constrained
    optimization. The hessian of the vector of constraints (both equality and inequality) and its tensor
    dot product with the lagrange multipliers are needed in forming the overall Hessian of the lagrangian function.
    
    Parameters
    -------------
    f : callable
        Real-valued vector function whose jacobian and hessian are to be approximated.
        Accepted output shapes are 0-d scalars and 1-d array_like. An exception is raised
        for higher dimensional outputs.
        
    x : array_like
        Real-valued vector on which to approximate the jacobian and hessian.
        
    h : int, float, or array_like, optional
        The stepsize to use for the jacobian and hessian approximation.
    
    order : {2, 4, 6, 8, 10}, optional
        The order of the central finite difference method to use. The mixed second-order
        partial derivatives of the Hessian are not approximated if the input vector x has only
        one component. Otherwise, they are efficiently approximated with minimum additional
        function evaluations. For instance, the efficient second-order approximation for the
        mixed partial derivatives can be found in wikipedia [1]. It only requires two (2) more
        function evaluation for each unique, off-diagonal entry. Taylor series expansion shows
        that the 2nd order Hessian approximation is exact for polynomials of degree three (3)
        or less, since its truncation error is proportional to the fourth derivatives.
        
        The other methods also require minimum additional function evaluations of
        4, 6, 8, and 10, respectively. They are derived using 2-d taylor series expansion in a
        Jupyter Lab notebook with the help of SymPy.
        
    H_form : string, {'default', 'blockmat', 'hypermat'}, optional
        The format of the 3-d hessian.
        
        'default' : the 3-d array has shape (m,n,n), where m and n are the lengths
        of the output and input vectors, respectively.
        
        'blockmat' : The hessian matrices are stacked row-by-row resulting in a 2-d array
        shape of (m*n,n).
        
        'hypermat' : The output is a (n,n,m) 3-d array where H[:,:,k] corresponds to the kth
        hessian matrix.
        
    zero_tol : float, optional
        The tolerance for considering if the entries of the jacobian and hessian are zero.
        Values with smaller magnitudes than this are replaced by zero.
        
    args : list or tuple, optional
        Additional parameters to pass to the function. Cannot be an empty list or tuple.
    
    
    Returns
    --------
    J : 1d or 2-d array
        The Jacobian approximation.
        
    H : 2d or 3-d array
        The Hessian approximation.
    
    
    References
    --------
    [1] Finite difference. https://en.wikipedia.org/wiki/Finite_difference
    [2] numderivative. https://help.scilab.org/docs/6.1.1/en_US/numderivative.html
    [3] numdifftools. https://numdifftools.readthedocs.io/en/latest/
    [4] Scilab. (2017). https://github.com/opencollab/scilab/blob/master/scilab/modules/differential_equations/macros/numderivative.sci
    """
    f, x, h, order, H_form, zero_tol, fx, n, m = validate_inputs(f, x, h, order, H_form, zero_tol, args)
    
    if order==2:
        stencil = np.arange(1,-2,-1)
        stencil_weights = np.array([
            [0.5, 0, -0.5],
            [1, -2, 1]
            ])
    elif order==4:
        stencil = np.arange(2,-3,-1)
        stencil_weights = np.array([
            [-1/12, 2/3, 0, -2/3, 1/12],
            [-1/12, 4/3, -5/2, 4/3, -1/12]
            ])
    elif order==6:
        stencil = np.arange(3,-4,-1)
        stencil_weights = np.array([
            [1/60, -3/20, 3/4, 0, -3/4, 3/20, -1/60],
            [1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90]
            ])
    elif order==8:
        stencil = np.arange(4,-5,-1)
        stencil_weights = np.array([
            [-1/280, 4/105, -1/5, 4/5, 0, -4/5, 1/5, -4/105, 1/280],
            [-1/560, 8/315, -1/5, 8/5, -205/72, 8/5, -1/5, 8/315, -1/560]
            ])
    else:
        stencil = np.arange(5,-6,-1)
        stencil_weights = np.array([
            [1/1260, -5/504, 5/84, -5/21, 5/6, 0, -5/6, 5/21, -5/84, 5/504, -1/1260],
            [1/3150, -5/1008, 5/126, -5/21, 5/3, -5269/1800, 5/3, -5/21, 5/126, -5/1008, 1/3150]
            ])
    # Scalar multi-variable case
    if m==0:
        J = np.zeros(n)
        H = np.zeros((n,n))
        for i in range(n):
            f_vals = np.array([
                fx if si==0 else f(x+si*h[i]) for si in stencil
                ])
            J[i] = stencil_weights[0] @ f_vals / h[i,i]
            H[i,i] = stencil_weights[1] @ f_vals / h[i,i]**2
        # Compute off-diagonal elements of Hessian
        # Exploit the symmetry of the Hessian
        if n>1:
            for i in range(n-1):
                for j in range(i+1,n):
                    r = h[i,i]/h[j,j]
                    f_vals = np.array([
                        fx if si==0 else f(x + si*(h[i]+h[j])) for si in stencil
                        ])
                    H[i,j] = stencil_weights[1] @ f_vals / (2*h[i,i]*h[j,j]) - (r**2*H[i,i] + H[j,j]) / (2*r)
                    H[j,i] = H[i,j]
        else:
            pass
    # Vector output case. Indexing is 3-d.
    else:
        J = np.zeros((m,n))
        H = np.zeros((m,n,n))
        for i in range(n):
            f_vals = np.array([
                fx if si==0 else f(x+si*h[i]) for si in stencil
                ])
            J[:,i] = stencil_weights[0] @ f_vals / h[i,i]
            H[:,i,i] = stencil_weights[1] @ f_vals / h[i,i]**2
        # Compute off-diagonal elements of Hessian
        # Exploit the symmetry of the Hessian
        if n>1:
            for i in range(n-1):
                for j in range(i+1,n):
                    r = h[i,i]/h[j,j]
                    f_vals = np.array([
                        fx if si==0 else f(x + si*(h[i]+h[j])) for si in stencil
                        ])
                    H[:,i,j] = stencil_weights[1] @ f_vals / (2*h[i,i]*h[j,j]) - (r**2*H[:,i,i] + H[:,j,j]) / (2*r)
                    H[:,j,i] = H[:,i,j]
        else:
            pass
    
        if H_form=='hypermat':
            H = H.T
        elif H_form=='blockmat':
            H = blockmat_reshape(H)
        else:
            pass
    
    # Clean out the output. Values smaller than zero_tol in magnitude
    # are considered zero.
    J = np.where(np.abs(J)<=zero_tol, 0, J)
    H = np.where(np.abs(H)<=zero_tol, 0, H)
    
    return J, H
