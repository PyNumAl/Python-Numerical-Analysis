from sympy import Function, symbols, ones, Matrix, linsolve, factorial
from numpy import median, asarray, diff
import numpy as np

class fdcc:
    """
    Finite Difference Coefficients Calculator (FDCC)
    
    Using the Taylor Series method, derive the corresponding finite difference expression, weights, and principal
    error term given a stencil and the desired derivative order.
    
    The taylor series method is the most common way for deriving finite difference coefficients in most 
    numerical analysis texts and resources. Thus, a derivation of the finite difference
    coefficients using taylor series is appropriate and fitting.
    
    
    IMPORTANT NOTE (11:29 AM, 2022-07-25):
    
        This calculator becomes numerically unstable for very high derivative orders. An example is as follows:
        
            >>> sol =  fdcc([0,1,2,3,4,5,6,7,8,9,10],10)
            >>> print(sol.finite_diff_weights)
            [14175/134231903, -141750/134231903, 637875/134231903, -1701000/134231903, 2976750/134231903, -3572100/134231903, 2976750/134231903, -1701000/134231903, 637875/134231903, -141750/134231903, 14175/134231903]
            
        Meanwhile, the output of the other FDCC using Lagrange Polynomials (https://github.com/PyNumAl/Python-Numerical-Analysis/blob/main/Numerical%20Differentiation/finite_diff_coeff_calculator.py)
        is much more readable despite being a little bit slower:
            
            >>> sol =  FDCC([0,1,2,3,4,5,6,7,8,9,10],10)
            >>> print(sol.finite_diff_weights)
            [1, -10, 45, -120, 210, -252, 210, -120, 45, -10, 1]
           
        Both calculators yield the same readable output for the 9-point forward finite difference formula for the 9th-derivative:
            
            >>> sol = fdcc([0,1,2,3,4,5,6,7,8,9],9)
            >>> print(sol.finite_diff_weights)
            [-1, 9, -36, 84, -126, 126, -84, 36, -9, 1]
            
    
    Parameters
    --------
    s : array_like
        Stencil points. Must be unique and strictly increasing.
    k : int
        The desired derivative order
    
    
    Returns
    --------
    finite_difference : SymPy expression
        The finite difference formula.
        
    finite_diff_weights : list
        The coefficients of the finite difference stencil from left to right.
        
    principal_error_term : SymPy expression
        The order of the method, proportional to the kth power of h.
    
    
    References
    --------
    [1] Taylor, Cameron R. (2016). Finite Difference Coefficients Calculator. [Online]. Available at: https://web.media.mit.edu/~crtaylor/calculator.html
    """
    def __init__(self, s, k):
        
        s = self._check_input(s,k)
        
        x, h = symbols('x,h',real=True)
        f = Function('f',real=True)
        
        xs = [si*h for si in s]
        fs = [f(x + xsi) for xsi in xs]

        n = len(s)
        if median(s)==0:
            terms = n+2
        else:
            terms = n+1
        A = ones(terms,n)
        for j in range(n):
            for i in range(1,terms):
                A[i,j] = s[j]**i / factorial(i)
        b = Matrix([0 if i!=k else 1 for i in range(n)])
        c = Matrix( linsolve((A[:n,:n],b)).args[0] )

        if median(s)==0:
            terms = n+2
        else:
            terms = n+1
        T = ones(terms,n)
        for j in range(n):
            for i in range(terms):
                if i==0:
                    T[i,j] = f(x)
                else:
                    T[i,j] = f(x).diff(x,i) * (xs[j])**i / factorial(i)
        cons = f(x).diff(x,k) - sum(T*c/h**k)
        
        self.finite_difference = ( c.dot(fs)/h**k ).cancel()
        self.finite_diff_weights = list(c)
        self.principal_error_term = cons
        
    def __repr__(self):
        attrs = ['finite_difference', 'finite_diff_weights', 'principal_error_term']
        m = max(map(len, attrs)) + 1
        return '\n'.join([a.rjust(m) + ': ' + repr(getattr(self,a)) for a in attrs])
    
    @staticmethod
    def _check_input(s,k):
        s = asarray(s)
        # s must be 1d
        if s.ndim!=1: raise ValueError('s must be a 1d input.')
        # Check for uniqueness & sortedness of s
        if np.any( diff(s) <=0 ): raise ValueError('Stencil Points must be unique and increasing.')
        # s must be greater than 1
        if s.size<2: raise ValueError('The number of stencil points must be at least 2.')
        
        # Check for scalar input to k
        if asarray(k).ndim!=0: raise ValueError('k must be a scalar.')
        # k must be an integer, neither float nor complex
        if type(k)!=int: raise ValueError('Derivative input must be an integer.')
        # Check for k>=len(s)
        if k>=s.size: raise ValueError('Derivative order must be strictly less than the number of stencil points.')
        # k cannot be k<=0
        if k<=0: raise ValueError('Derivative order must be at least one.')
        return s
