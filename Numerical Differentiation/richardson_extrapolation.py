import numpy as np
from warnings import warn

eps = np.finfo(float).eps

class numdiff:
    """
    Numerically differentiate a scalar real-valued function.
    
    This class provides for approximating the derivatives of a scalar real-valued
    function via richardson extrapolation of low-order (1st and 2nd order)
    finite difference formulas to try achieve and satisy the given tolerances.
    Up to the tenth derivative can be approximated.
    
    The approximations are extrapolated up to 12th order if needed to satisfy the
    tolerances, though the approximated value may not necessarily of the same order of
    accuracy due to round-off error if the perturbation step h is "too small".
    Whether the perturbation step is too small or not depends on the derivative order
    being approximated; the finite differences for higher derivative orders are divided
    by higher powers of the stepsize h. Hence, either a larger stepsize must 
    be used to avoid round-off error - which can be
    problematic when the function is rapidly changing around the point of interest.
    Or set larger values for the tolerances.
    
    In either case, numerical instability is more prominent for higher order derivatives
    (derivative orders 6 and greater) and they can only be approximated more roughly
    than lower order derivatives.
    
    Parameters
    --------
    fun : callable
        The scalar real-valued function whose derivative is to be approximated.
    
    Methods
    --------
    __call__(x, h, k, mode, rtol, atol)
    
        Parameters
        --------
        x : array_like
            Points to obtain the derivative approximation at.
            
        h : float, optional
            The finite step to use in approximating the derivatives.
            Too small of a stepsize will result in round-off errors
            dominating the approximated values.
        
        k : float or array_like, optional
            Desired derivative order. Default is one. Range of valid values is
            from 1 to 10.
        
        mode : {'central', 'forward'}, optional
            The finite-difference method to use. Forward finite differences
            are less accurate (1st order) than central finite differences (2nd order)
            but they are useful when the function can only be evaluated one-sidedly
            to obtain sensible derivative approximations e.g. functions like
            1/x, log(x), etc.
            
            If backward differences are desired, simply pass in a negative value for
            the stepsize h and set mode equal to 'forward'.
            
        rtol, atol : float, optional
            The relative and absolute tolerances. Default values are 1e-3
            and 1e-6 respectively.
    """
    def __init__(self,fun):
        self.fun = fun
        self.fundiff = np.vectorize(self._extrapolate, excluded=[1, 3, 4, 5])
        
    def __call__(self, x, h=.1, k=1, mode='central', rtol=1e-3, atol=1e-6):
        self._validate_tol(rtol,atol)
        self._validate_input(self.fun, x, h, k, mode)
        return self.fundiff(x, h, k, mode, rtol, atol)
    
    @staticmethod
    def _validate_input(fun, x, h, k, mode):
        x = np.asarray(x)
        k = np.asarray(k)
        if x.dtype not in [int,float]:
            raise TypeError('x must be a real-valued input i.e. int or float.')
        elif np.asarray(fun(x)).dtype not in [int,float]:
            raise TypeError('fun must be a scalar real-valued function.')
        elif type(h)!=float:
            raise TypeError('Stepsize h must be a float.')
        elif h==0:
            raise ValueError('Stepsize h cannot be zero.')
        elif k.dtype!=int:
            raise TypeError('Invalid type for k. k must be an integer/s.')
        elif np.any(k<1) or np.any(k>10):
            raise ValueError('Invalid value/s for k. Range of valid values for k is from 1 to 10.')
        elif mode not in ['central', 'forward']:
            raise Exception(f"Argument mode can only be {'central'} or {'forward'} ")
        else:
            pass
        return None
    
    @staticmethod
    def _validate_tol(rtol, atol):
        if type(rtol)!=float or type(atol)!=float:
            raise TypeError('rtol and atol must be floats.')
        elif rtol<0 or atol<0:
            raise ValueError('rtol and atol cannot be negative.')
        elif rtol<1e3*eps:
            warn(f'rtol value too small, setting to {1e3*eps}')
            rtol = 1e3*eps
        else:
            pass
        return None
    
    @staticmethod
    def _get_weights(k,mode):
        if mode=='central':
            weights = [ [-1/2, 0, 1/2],
                       [1, -2, 1],
                       [-1/2, 1, 0, -1, 1/2],
                       [1, -4, 6, -4, 1],
                       [-1/2, 2, -5/2, 0, 5/2, -2, 1/2],
                       [1, -6, 15, -20, 15, -6, 1],
                       [-1/2, 3, -7, 7, 0, -7, 7, -3, 1/2],
                       [1, -8, 28, -56, 70, -56, 28, -8, 1],
                       [-1/2, 4, -27/2, 24, -21, 0, 21, -24, 27/2, -4, 1/2],
                       [1, -10, 45, -120, 210, -252, 210, -120, 45, -10, 1]
                       ]
        else:
            weights = [
                [-1, 1],
                [1, -2, 1],
                [-1, 3, -3, 1],
                [1, -4, 6, -4, 1],
                [-1, 5, -10, 10, -5, 1],
                [1, -6, 15, -20, 15, -6, 1],
                [-1, 7, -21, 35, -35, 21, -7, 1],
                [1, -8, 28, -56, 70, -56, 28, -8, 1],
                [-1, 9, -36, 84, -126, 126, -84, 36, -9, 1],
                [1, -10, 45, -120, 210, -252, 210, -120, 45, -10, 1]
                ]
        return weights[k-1]
    
    @staticmethod
    def _feval_stencil(fun,x,h,weights,mode):
        n = len(weights)
        if mode=='central':
            num_each_side = int((n-1)/2)
            funvals = [fun(x+i*h) for i in range(-num_each_side, num_each_side+1, 1)]
        else:
            funvals = [fun(x+i*h) for i in range(n)]
        return funvals
    
    def _approx_derivative(self, x, h, k, mode):
        weights = self._get_weights(k,mode)
        funvals = self._feval_stencil(self.fun, x, h, weights, mode)
        return np.dot(weights,funvals) / h**k
    
    def _extrapolate(self, x, h, k, mode, rtol, atol):
        func = lambda x, h: self._approx_derivative(x, h, k, mode)
        dim = {'central':6
               ,
               'forward':12
               }[mode]
        T = np.zeros((dim,dim))
        T[0,0] = func(x,h)
        
        for i in range(1,dim):
            T[i,0] = func(x,h/2**i)
            for j in range(i):
                c = {'central':2**(2+2*j)
                     ,
                     'forward':2**(1+j)
                     }[mode]
                T[i,j+1] = ( c*T[i,j] - T[i-1,j] ) / (c-1)
                if np.allclose(T[i,j], T[i,j+1], rtol, atol):
                    self.table = T[:i+1,:j+2]
                    return T[i,j+1]
        else:
            self.table = T
            return T[-1,-1]
