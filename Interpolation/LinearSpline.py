import numpy as np
from numba import guvectorize

class LinearSpline:
    """
    Linear spline approximation of data.
    
    Parameters : x : array_like, shape (n,)
                     1-D array containing values of the independent variable. Values must be strictly unique and increasing.
        
                y : array_like, shape (n,) or (n,k)
                    Array containing values of the dependent variable. Can be 1d or 2d. If 2d, axis 0 ---
                    the number of rows --- must have the same length n as x while axis 1, the number of columns,
                    corresponds to the number of y components.
    """
    def __init__(self, x, y):
        
        x = np.asfarray(x)
        y = np.asfarray(y)
        
        # input checks for x
        if x.ndim!=1:
            raise ValueError('x must be 1d.')
        elif x.size<2:
            raise ValueError('Number of samples must be at least 2.')
        elif np.any(x[1:]-x[:-1]<=0):
            raise ValueError('x samples must be strictly unique and increasing.')
        else:
            pass
        N = x.size-1
        
        # input checks for y
        if y.ndim!=1 and y.ndim!=2:
            raise ValueError('y can either be 1d or 2d.')
        elif y.ndim==1:
            if y.size!=x.size:
                raise ValueError('x and y samples must have the same length for 1d input.')
            else:
                y=y.reshape((y.size,1))
        elif y.ndim==2:
            dim1, dim2 = y.shape
            if dim1 < 2:
                raise ValueError('Number of samples must be at least 2.')
            else:
                pass
        else:
            pass
        
        h = np.diff(x)
        h = np.full((y.shape[1],h.size), h).T
        
        a0 = y[:-1]
        a1 = (y[1:] - y[:-1]) / h
        
        self.x = x
        self.y = y
        self.a0 = a0
        self.a1 = a1
        
        @guvectorize(['void(f8, i8, f8[:], f8[:,:], f8[:,:], f8[:])'], '(), (), (N), (n,k), (n,k)->(k)')
        def eval_spl(u, n, x, a0, a1, out):
            # find spline location
            if u<=x[0]:
                i=0        
            elif u>=x[N]:
                i=N-1
            else:
                i=np.where(x>u)[0][0]-1
            if n==0:
                S=a0[i] + a1[i]*(u - x[i])
            elif n==1:
                S=a1[i]
            else:
                S=np.zeros_like(a1[i])
            out[:]=S
            return None
        
        @guvectorize(['void(f8, f8[:], f8[:,:], f8[:])'], '(),(N),(n,k)->(N)', nopython=True)
        def shift_x(u, x, y, out):
            if np.all( np.abs( y[0] - y[-1] ) <= 1e-8 )!=True:
                raise ValueError('Endpoint y values be the same for periodic interpolation.')
            period = x[-1] - x[0]
            if u<x[0]:
                shifted_x = x - np.ceil( (x[0] - u)/period )*period
            elif u>x[-1]:
                shifted_x = x + np.ceil( (u - x[-1])/period )*period
            else:
                shifted_x = x
            out[:] = shifted_x
            
            return None
        
        self.eval_spl = eval_spl
        self.shift_x = shift_x
        
    def lerp(self, u, n=0):
        """
        Method for piecewise linear approximation.
        
        Parameters : u : array_like
                         The points to interpolate. If outside the boundaries, it is extrapolated linearly.
        
                     n : int, optional
                         Order of derivative to evaluate. Must be non-negative.
        
        Returns :    y : array_like
                         Interpolated values.
        """
        u, n = self._check_input(u,n)
        return self.eval_spl(u, n, self.x, self.a0, self.a1)
    
    def perp(self, u, n=0):
        """
        Method for periodic piecewise linear approximation.
        
        Parameters : u : array_like
                         The points to interpolate. If outside the boundaries, periodic extrapolation is done.
        
                     n : int, optional
                         Order of derivative to evaluate. Must be non-negative.
        
        Returns :    y : array_like
                         Periodically Interpolated values.
        """
        u, n = self._check_input(u,n)
        shifted_x = self.shift_x(u, self.x, self.y)
        return self.eval_spl(u, n, shifted_x, self.a0, self.a1)
    
    @staticmethod
    def _check_input(u, n):
        u=np.asfarray(u)
        n=np.asarray(n,int)
        if n.dtype!=int or np.any(n<0):
            raise ValueError('n must be a nonnegative integer')
        else:
            pass
        return u, n