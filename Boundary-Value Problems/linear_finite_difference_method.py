import numpy as np
import numpy.linalg as LA
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.interpolate import BPoly

class LFDM:
    """
    Linear Finite Difference Method for 2nd Order Linear Boundary-Value Problems
    
    Solve a two-point linear boundary-value problem. Complex integration is not supported.
    
    The linear ODE that's discretized and solved is of the following form:
        
        a2(x)*ypp + a1(x)*yp + a0(x)*y = f(x)
        
        with boundary conditions,
        
        wa[0]*y0 + wa[1]*yp0 = wa[2]
        wb[0]*y_n + wb[1]*yp_n = wb[2].
    
    Parameters
    ----------
    a2 : callable, int or float
        The constant or variable coefficient of the 2nd derivative term.
    a1 : callable, int, or float
        The constant or variable coefficient of the 1st derivative term.
    a0 : callable, int or float
        The constant or variable coefficient of the 0th derivative term i.e. y itself.
    f : callable, inf, or float
        The right-hand-side forcing function.
    a : int or float
        The left edgepoint. Must be less than argument b.
    b : int or float
        The right edgepoint. Must be greater than argument a.
    wa : array_like of length 3
        The left boundary condition. wa[0] and wa[1] are the coefficients of
        y0 and yp0, respectively, while wa[2] is the right-hand side constant.
    wb : array_like of length 3
        The right boundary condition. wb[0] and wb[1] are the coefficients of
        y_n and yp_n, respectively, while wb[2] is the right-hand side constant.
    N : int
        The number of steps/meshes/subintervals.
    order : {2, 4, 6, or 8}, optional
        The desired accuracy of the solution. Default is 2nd order. Higher order
        accuracy is achieved by richardson extrapolation.
    
    Returns
    -------
    Bunch object with the following fields defined:
        sol : BPoly
            Found solution for y as `scipy.interpolate.BPoly` instance, a C2 continuous
            quintic spline.
        x : ndarray, shape (m,)
            Nodes of the mesh.
        y : ndarray, shape (m,)
            Solution values at the mesh nodes.
        dy : ndarray, shape (m,)
            1st derivative values at the mesh nodes.
        d2y : ndarray, shape (m,)
            2nd derivative values at the mesh nodes.
        error_norm : array_like of length 3
            The error norm (maximum norm) of the solution values, y, 
            and its 1st two derivatives, dy and d2y. It is obtained via
            richardson extrapolation. None for order=2.
    """
    def __init__(self, a2, a1, a0, f, a, b, wa, wb, N, order=2):
        
        a2, a1, a0, f, a, b, wa, wb, N, order = self._validate_input(a2, a1, a0, f, a, b, wa, wb, N, order)
        
        Y = self._generate_and_solve(a2, a1, a0, f, a, b, wa, wb, N)
        y, dy, d2y = Y.T[[0,1,2]]
        
        if order==2:
            self.error_norm = 'Not available for order 2.'
        
        elif order==4:
            Y2 = self._generate_and_solve(a2, a1, a0, f, a, b, wa, wb, 2*N)
            Y3 = ( 4*Y2[::2] - Y ) / 3
            self.error_norm = LA.norm(Y3 - Y2[::2], np.inf, axis=0)
            y[:], dy[:], d2y[:] = Y3.T[ [0, 1, 2] ]
            Y[:] = Y3
        
        elif order==6:
            Y2 = self._generate_and_solve(a2, a1, a0, f, a, b, wa, wb, 2*N)
            Y3 = self._generate_and_solve(a2, a1, a0, f, a, b, wa, wb, 4*N)
            Y4 = ( 4*Y2[::2] - Y ) / 3  # from 2nd order to 4th order
            Y5 = ( 4*Y3[::2] - Y2 ) / 3 # from 2nd order to 4th order
            Y6 = ( 16*Y5[::2] - Y4) / 15 # from 4th order to 6th order
            self.error_norm = LA.norm(Y6 - Y5[::2], np.inf, axis=0)
            y[:], dy[:], d2y[:] = Y6.T[ [0, 1, 2] ]
            Y[:] = Y6
        
        else:
            Y2 = self._generate_and_solve(a2, a1, a0, f, a, b, wa, wb, 2*N)
            Y3 = self._generate_and_solve(a2, a1, a0, f, a, b, wa, wb, 4*N)
            Y4 = self._generate_and_solve(a2, a1, a0, f, a, b, wa, wb, 8*N)
            
            Y5 = ( 4*Y2[::2] - Y ) / 3  # from 2nd order to 4th order
            Y6 = ( 4*Y3[::2] - Y2 ) / 3  # from 2nd order to 4th order
            Y7 = ( 4*Y4[::2] - Y3 ) / 3  # from 2nd order to 4th order
            
            Y8 = ( 16*Y6[::2] - Y5) / 15 # from 4th order to 6th order
            Y9 = ( 16*Y7[::2] - Y6) / 15 # from 4th order to 6th order
            
            Y10 = ( 64*Y9[::2] - Y8) / 63 # from 6th order to 8th order
            
            self.error_norm = LA.norm(Y10 - Y9[::2], np.inf, axis=0)
            y[:], dy[:], d2y[:] = Y10.T[ [0, 1, 2] ]
            Y[:] = Y10
        
        self.x = np.linspace(a, b, N+1)
        self.y = y
        self.dy = dy
        self.d2y = d2y
        sol = BPoly.from_derivatives(self.x, Y)
        self.sol = sol
    
    def __repr__(self):
        attrs = ['sol', 'x', 'y', 'dy', 'd2y', 'error_norm']
        m = max( map(len, attrs) ) + 1
        return '\n'.join([ a.rjust(m) + ': ' + repr(getattr(self,a)) for a in attrs ])
    
    @staticmethod
    def _validate_input(a2, a1, a0, f, a, b, wa, wb, N, order):
        # Check coefficients and forcing function.
        if callable(a2) is not True:
            a2=lambda x,a2=a2: np.full(x.size,a2)
        if callable(a1) is not True:
            a1=lambda x,a1=a1: np.full(x.size,a1)
        if callable(a0) is not True:
            a0=lambda x,a0=a0: np.full(x.size,a0)
        if callable(f) is not True:
            f=lambda x,f=f: np.full(x.size,f)
        # Check the a and b
        if type(a)!=int and type(a)!=float:
            raise ValueError('a must be an int or float.')
        elif type(b)!=int and type(b)!=float:
            raise ValueError('b must be an int or float.')
        elif (a<b)==False:
            raise ValueError('b must be strictly greater than a.')
        else:
            pass
        # Check wa and wb
        wa = np.asfarray(wa)
        wb = np.asfarray(wb)
        if wa.size!=3 or wb.size!=3:
            raise ValueError('wa and wb must be of length 3.')
        elif np.all( np.isclose(wa[:2], 0., atol=1e-10) ):
            raise ValueError('Invalid left boundary condition. wa[0] and wa[1] cannot be both zero or nearly zero.')
        elif np.all( np.isclose(wb[:2], 0., atol=1e-10) ):
            raise ValueError('Invalid right boundary condition. wb[0] and wb[1] cannot be both zero or nearly zero.')
        else:
            pass
        # Check number of meshes
        if type(N) is not int:
            raise ValueError('N should be an interger.')
        elif N<2:
            raise ValueError('The number of meshes must be at least 2.')
        else:
            pass
        
        if order!=2 and order!=4 and order!=6 and order!=8:
            raise ValueError('order can only be 2, 4, 6, or 8 only.')
        else:
            pass
        
        return a2, a1, a0, f, a, b, wa, wb, N, order
        
    def _get_derivatives(self, y, N, h):
        """
        Obtain the first and second derivatives of y.
        The second derivative is obtained using the appropriate 2nd-order accurate
        finte difference formulas, not by recursive use of np.gradient on the 
        1st derivative since it leads to less acurate results.
        """
        dy = np.gradient(y, h, edge_order=2)
        d2y = np.zeros_like(y)
        if y.size < 4:
            d2y[:] = 1/h**2 * (y[0] - 2*y[1] + y[2])
        else:
            d2y[1:-1] = 1/h**2 * ( y[0:N-1] - 2*y[1:N] + y[2:N+1] )
            d2y[0] = 1/h**2 * np.matmul([2, -5, 4, -1], y[:4])
            d2y[-1] = 1/h**2 * np.matmul([-1, 4, -5, 2], y[-4:])
        return dy, d2y
    
    def _generate_and_solve(self, a2, a1, a0, f, a, b, wa, wb, N):
        """
        Helper function for repeatedly solving the boundary-value problem
        on finer meshes.
        Used for richardson extrapolation to achieve higher-order-accurate results.
        """
        w1,w2,w3=wa
        w4,w5,w6=wb
        x,h=np.linspace(a ,b, N+1,retstep=True)
        
        a2=a2(x)
        a1=a1(x)
        a0=a0(x)
        f=f(x)
        
        b=2*h**2*f[1:N]
        p=2*a2[1:N] - h*a1[1:N]
        q=-4*a2[1:N] + 2*h**2*a0[1:N]
        r=2*a2[1:N] + h*a1[1:N]
        
        rhs=np.hstack((2*h*w3, b, 2*h*w6))
        
        k0=np.hstack((2*h*w1 - 3*w2, q, 2*h*w4 + 3*w5))
        k1=np.hstack((4*w2, r))
        k2=np.hstack((-w2, np.zeros(N-2) ))
        kn1=np.hstack((p, -4*w5))
        kn2=np.hstack(( np.zeros(N-2), w5))
        A=diags([kn2,kn1,k0,k1,k2], [-2,-1,0,1,2]).tocsc()
        
        y=spsolve(A,rhs)
        dy, d2y = self._get_derivatives(y, N, h)
        Y = np.array([y, dy, d2y]).T
        return Y