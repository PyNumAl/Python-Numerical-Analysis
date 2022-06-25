import numpy as np

class derivatives:
    """
    Class providing for methods to approximate the jacobian, gradient, or hessian
    using 1st-order, 2nd-order, or 4th-order-accurate finite differences,
    inspired by the numderivative function of the SciLab numerical programming language.
    https://help.scilab.org/docs/6.1.1/en_US/numderivative.html
    https://github.com/opencollab/scilab/blob/master/scilab/modules/differential_equations/macros/numderivative.sci
    """
    def __init__(self, f):
        self.f = f
        
    def fjac(self, x, h=None, order=2, args=None):
        """
        Jacobian approximation
        
        x : Vector of independent variables
        h : The stepsize to use for computing the derivatives.
        order : The order of the finite difference approximations to use.
        args : Additional parameter inputs to function f
        """
        f, x, h, order, n, fx, H = self._validate_input(self.f, x, h, order, args, 1, 1)
        if x.dtype==complex:
            J=np.zeros((n,n), complex)
        else:
            J=np.zeros((n,n), float)
        for i in range(n):
            Hi=H[i]
            hi=h[i]
            if order==1:
                J[:,i]=1/hi * np.matmul([-1, 1], [fx, f(x+Hi)])
            elif order==2:
                J[:,i]=1/hi * np.matmul([-.5, 0., .5], [f(x-Hi), fx, f(x+Hi)])
            else:
                J[:,i]=1/hi * np.matmul([1/12, -2/3, 0., 2/3, -1/12], [f(x-2*Hi), f(x-Hi), fx, f(x+Hi), f(x+2*Hi)])
        return J
    
    def fgrad(self, x, h=None, order=2, args=None):
        """
        Gradient Approximation
        
        x : Vector of independent variables
        h : The stepsize to use for computing the derivatives.
        order : The order of the finite difference approximations to use.
        args : Additional parameter inputs to function f
        """
        f, x, h, order, n, fx, H = self._validate_input(self.f, x, h, order, args, 0, 1)
        if x.dtype==complex:
            G=np.zeros(n, complex)
        else:
            G=np.zeros(n, float)
        for i in range(n):
            Hi=H[i]
            hi=h[i]
            if order==1:
                G[i]=1/hi * np.matmul([-1, 1], [fx, f(x+Hi)])
            elif order==2:
                G[i]=1/hi* np.matmul([-.5, 0., .5], [f(x-Hi), fx, f(x+Hi)])
            else:
                G[i]=1/hi * np.matmul([1/12, -2/3, 0., 2/3, -1/12], [f(x-2*Hi), f(x-Hi), fx, f(x+Hi), f(x+2*Hi)])
        return G
    
    def fhess(self, x, h=None, order=2, args=None):
        """
        Hessian approximation
        
        x : Vector of independent variables
        h : The stepsize to use for computing the derivatives.
        order : The order of the finite difference approximations to use.
        args : Additional parameter inputs to function f
        """
        f, x, h, order, n, fx, H = self._validate_input(self.f, x, h, order, args, 0, 2)
        if x.dtype is complex:
            Hessian=np.zeros((n,n), complex)
        else:
            Hessian=np.zeros((n,n), float)
        for i in range(n):
            Hi=H[i]
            hi=h[i]
            for j in range(n):
                Hj=H[j]
                hj=h[j]
                if order==1:
                    diff1=f(x + Hj + Hi) - f(x + Hj)
                    diff2=f(x + Hi) - fx
                    Hessian[i,j]=1/(hj*hi) * ( diff1 - diff2 )
                elif order==2:
                    diff1=f(x + Hj + Hi) - f(x + Hj - Hi)
                    diff2=f(x - Hj + Hi) - f(x - Hj - Hi)
                    Hessian[i,j]=1/(4*hj*hi) * ( diff1 - diff2 )
                else:
                    diff1=f(x - 2*Hj - 2*Hi) - 8*f(x - 2*Hj - Hi) + 8*f(x - 2*Hj + Hi) - f(x - 2*Hj + 2*Hi)
                    diff2=f(x - Hj - 2*Hi) - 8*f(x - Hj - Hi) + 8*f(x - Hj + Hi) - f(x - Hj + 2*Hi)
                    diff3=f(x + Hj - 2*Hi) - 8*f(x + Hj - Hi) + 8*f(x + Hj + Hi) - f(x + Hj + 2*Hi)
                    diff4=f(x + 2*Hj - 2*Hi) - 8*f(x + 2*Hj - Hi) + 8*f(x + 2*Hj + Hi) - f(x + 2*Hj + 2*Hi)
                    Hessian[i,j]=1/(144*hj*hi) * (diff1 - 8*diff2 + 8*diff3 - diff4)
        return Hessian
    
    @staticmethod
    def _validate_input(f, x, h, order, args, out, deriv_order):
        """
        Validate the inputs.
        f : The input scalar/vector function
        x : Vector of independent variables
        h : The stepsize to use for computing the derivatives.
        order : The order of the finite difference approximations to use.
        args : Additional parameter inputs to function f
        out : {0,1} 0 means that the output of f is a scalar i.e. f is a scalar
            multivariable function while 1 means indicates that f is a vector function.
        deriv_order : {1,2} 1 for jacobian/gradient approximation while 2 for the hessian approximation
        """
        if args is not None:
            # check args if instance of list or tuple
            if isinstance(args, tuple) or isinstance(args, list):
                f=lambda x,f=f: np.asarray(f(x, *args))
        else:
            f=lambda x,f=f: np.asarray(f(x))
        
        x=np.asarray(x)
        if x.ndim!=1:
            raise ValueError('x must be a 1d array-like input.')
        n=x.size
        
        fx=f(x)
        if out==1:
            if fx.ndim!=1:
                raise ValueError('Output of f must be 1d array-like.')
            elif fx.size!=n:
                raise ValueError('Output of f must have the same length as x.')
            else:
                pass
        else:
            if fx.ndim!=0:
                raise ValueError('Outout of f must be a scalar.')
            else:
                pass
        
        if order!=1 and order!=2 and order!=4:
            raise ValueError('order can only be 1, 2, or 4.')
        if h is not None:
            h=np.asfarray(h)
            hdim=h.ndim
            if hdim!=0 and hdim!=1:
                raise ValueError('h can either be a scalar or 1d array-like.')
            elif hdim==0:
                h=np.full(n, h, float)
            elif hdim==1 and h.size!=x.size:
                raise ValueError('h must have the same length as x.')
            else:
                pass
        else:
            exponent = 1/(order + deriv_order)
            hdefault = (2.22e-16)**exponent
            h = (np.abs(x) + 1)*hdefault
        H = np.diag(h)
        
        return f, x, h, order, n, fx, H