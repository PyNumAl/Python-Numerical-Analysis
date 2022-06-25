import sympy as sp
import numpy as np

class FDCC:
    """
    Symbolic Finite Difference Coefficients Calculator
    Uses lagrange polynomials to calculate the finite difference formula,
    finite difference coefficients, and the leading error term 
    (and thus the order of the finite difference approximation).
    
    Inspired by the Finite Difference Coefficients Calculator of Cameron Taylor
    https://web.media.mit.edu/~crtaylor/calculator.html
    """
    def __init__(self, s, d):
        
        s = self._check_input(s,d)
        
        a,x,h=sp.symbols(r'a,x,h',real=True)
        f=sp.Function('f')
        Dfa=f(x).diff(x,d).subs(x,a)
    
        xp=[si*h for si in s]
        yp=[f(a+xpi) for xpi in xp]

        Lx=0
        for i,ypi in enumerate(yp):
            xpi=xp[i]
            basis=1
            for j,xpj in enumerate(xp):
                if j!=i:
                    basis=basis * ( x - xpj ) / ( xpi - xpj )
            Lx+=ypi*basis
        
        # One can obtain the finite difference formula by differentiating the
        # resulting Lagrange Polynomial d times as follows:
        #    fdf=Lx.diff(x,d).subs(x,0).cancel()
        # But this is particularly slow especially when involving 
        # more stencil points and higher derivatives.
        
        # The quite faster way is to expand the polynomial and obtain the 
        # coefficient expression of x**d and then multiply the expression 
        # by the constant, d! (factorial of d).
        fdf=( Lx.expand().coeff(x,d)*sp.factorial(d) ).cancel()
        fdc=[ (h**d*fdf).expand().coeff(ypi,1) for i,ypi in enumerate(yp) ]

        # Determine if the stencil is centered at zero.
        # The stencil is centered at zero if the median is zero.
        # This is true whether for odd or even number of stencils.
        # If the stencil is centered at zero, the error terms only consist
        # of even powers of h. Hence, the number of terms in the taylor series
        # expansion must be two (2) more than the number of stencil points in order
        # to obtain the leading error term.
        
        # Otherwise, for stencils not centered at zero, the error terms consist
        # of both odd and even powers of h and the number of terms needed is
        # only the stencil size plus one (1).
        if np.median(s)==0:
            terms=s.size+2
        else:
            terms=s.size+1
    
        # Taylor Series Expansion
        TSE=sp.zeros(s.size, terms)
        for i in range(s.size):
            xpi=xp[i]
            for j in range(terms):
                TSE[i,j]=f(x).diff(x,j).subs(x,a) * (xpi)**j / sp.factorial(j)
        
        # Get Leading Error Term
        rhsv=sp.Matrix(fdc).T * TSE / h**d
        rhs=sum(rhsv)
        LET=-(rhs-Dfa)
        
        self.finite_difference = fdf
        self.coefficients = fdc
        self.leading_error_term = LET
    
    def __repr__(self):
        attrs = ['finite_difference', 'coefficients', 'leading_error_term']
        m = max( map(len, attrs) ) + 1
        return '\n'.join([ a.rjust(m) + ': ' + repr( getattr(self,a) ) for a in attrs ])
    
    @staticmethod
    def _check_input(s,d):
        s=np.asarray(s)
        # s must be 1d
        if s.ndim!=1: raise ValueError('s must be a 1d input.')
        # Check for uniqueness & sortedness of s
        if np.any( np.diff(s) <=0 ): raise ValueError('Stencil Points must be unique and increasing.')
        # s must be greater than 1
        if s.size<2: raise ValueError('The number of stencil points must be at least 2.')
        
        # Check for scalar input to d
        if np.asarray(d).ndim!=0: raise ValueError('d must be a scalar.')
        # d must be an integer, neither float nor complex
        if type(d)!=int: raise ValueError('Derivative input must be an integer.')
        # Check for d>=len(s)
        if d>=s.size: raise ValueError('Derivative order must be strictly less than the number of stencil points.')
        # d cannot be d<=0
        if d<=0: raise ValueError('Derivative order must be at least one.')
        return s