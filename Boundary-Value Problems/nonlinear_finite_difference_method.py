import numpy as np
import numpy.linalg as LA
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from scipy.interpolate import BPoly
from warnings import warn

class NFDM:
    """
    Nonlinear Finite Difference Method for 2nd Order Nonlinear Boundary-Value Problems
    
    Solve a 2nd-order nonlinear boundary-value problem of the form:
        
        d2y / dx2 = f(x, y, dy/dx)
        
        with boundary conditions
        
        wa[0]*y_0 + wa[1]*yp_0 = wa[2]
        wb[0]*y_n + wb[1]*yp_n = wb[2]
    
    Complex integration is not supported.
    
    
    Parameters
    ----------
    f : callable
        Right-hand side of the system, d2y / dx2 = f. The calling signature is ``f(x, y, yp)``.
        Here `x` is the independent variable and `y` and `yp` is the dependent variable and its 1st
        derivative respectively. If the all arguments `x`, `y`, and `yp` are scalars, then the output of
        is a scalar. Basically, f is scalar multi-variable function of `x`, `y`, and `yp`.
    y : 1d array_like
        Required initial guess for the solution values, including the boundary values y_0 and y_n.
    a : int or float
        Left boundary coordinate. Must be strictly less than b.
    b : int or float
        Right boundary coordinate. Must be strictly greater than a.
    fy : callable, optional
        The partial derivative of f with respect to `y`. Its calling signature is ``fy(x, y, yp)``,
        same as for f. If not given, then it is approximated via centered finite-differences.
    fyp : callable, optional
        The partial derivative of f with respect to `yp`. Its calling signature is ``fyp(x, y, yp)``,
        same as for f. If not given, then it is approximated via centered finite-differences.
    order : {2, 4, 6, 8}, optional
        The desired order of the solution. Richardson extrapolation is done to achieve higher order accuracy.
    ftol : float
        Residual norm tolerance stopping criterion for the sparse newton iterations.
    ytol : float
        Absolute error norm tolerance for the solution values.
    rtol : float
        Relative error norm tolerance for the solution values.
    maxiter : int
        Maximum number of newton iterations. If none, then the default is 100 times the number of unknowns i.e. the size of y.
    
    
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
        res_norm : float
            The (maximum) norm of the residuals.
        niter : int
            The total number of iterations including those made during extrapolation.
        status : int
            Reason for algorithm termination.:
                
                * 0: The algorithm converged to the desired accuracy.
                * 1: The stopping criteria aren't satisfied within the `maxiter` iterations.
                * 2: A singular Jacobian is encountered during solving of the nonlinear equations.
                  system.
        
        
        message : string
            Verbal description of the termination reason.
        success : bool
            True if the algorithm converged to the desired accuracy (``status=0``).
    
    
    References
    ----------
    .. [1] R. L. Burden and J. D. Faires, "Finite-Difference Methods for Nonlinear Problems", in Numerical Analysis, 9th ed,
            ch. 11, pp. 691-696. Accessed: Jun. 28, 2022. [Online]. Available: https://faculty.ksu.edu.sa/sites/default/files/numerical_analysis_9th.pdf
    .. [2] Kevin Mooney, United States. Nonlinear Differential Equations Using Finite Differences: Can we Use Sparse Matrices?. (May 10, 2021). Accessed: June 28, 2022. [Online Video]. Available: https://www.youtube.com/watch?v=JYxUh5h_g-M
    """
    def __init__(self, f, y, a, b, wa, wb, fy=None, fyp=None, order=2, ftol=0.0, ytol=0.0, rtol=1e-10, maxiter=None):
        
        f, y, a, b, wa, wb, fy, fyp, order, ftol, ytol, rtol, maxiter = self._validate_input(f, y, a, b, wa, wb, fy, fyp, order, ftol, ytol, rtol, maxiter)
        
        N=y.size-1
        x,h=np.linspace(a, b, N+1, retstep=True)
        w1,w2,w3=wa
        w4,w5,w6=wb
        
        self.N = N
        self.h = h
        self.x = x
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.w5 = w5
        self.w6 = w6
        self.f = f
        self.fy = fy
        self.fyp = fyp
        self.ftol = ftol
        self.ytol = ytol
        self.rtol = rtol
        self.maxiter = maxiter
        self.niter = 0
        
        y[:] = self.sparse_newton(self.F, y, self.J, ftol, ytol, rtol, maxiter, [N, h, x])
        dy, d2y = self.get_derivatives(y, N, h)
        Y = np.array([y, dy, d2y]).T
        
        if order==2:
            self.error_norm = 'Not available for order 2.'
            
        elif order==4:
            Y2 = self.generate_and_solve(2*N, y)
            Y3 = ( 4*Y2[::2] - Y ) / 3
            self.error_norm = LA.norm(Y3 - Y2[::2], np.inf, axis=0)
            y[:], dy[:], d2y[:] = Y3.T[ [0, 1, 2] ]
            Y[:] = Y3
            
        elif order==6:
            Y2 = self.generate_and_solve(2*N, y)
            Y3 = self.generate_and_solve(4*N, y)
            Y4 = ( 4*Y2[::2] - Y ) / 3  # from 2nd order to 4th order
            Y5 = ( 4*Y3[::2] - Y2 ) / 3 # from 2nd order to 4th order
            Y6 = ( 16*Y5[::2] - Y4) / 15 # from 4th order to 6th order
            self.error_norm = LA.norm(Y6 - Y5[::2], np.inf, axis=0)
            y[:], dy[:], d2y[:] = Y6.T[ [0, 1, 2] ]
            Y[:] = Y6
            
        else:
            Y2 = self.generate_and_solve(2*N, y)
            Y3 = self.generate_and_solve(4*N, y)
            Y4 = self.generate_and_solve(8*N, y)
            
            Y5 = ( 4*Y2[::2] - Y ) / 3  # from 2nd order to 4th order
            Y6 = ( 4*Y3[::2] - Y2 ) / 3  # from 2nd order to 4th order
            Y7 = ( 4*Y4[::2] - Y3 ) / 3  # from 2nd order to 4th order
            
            Y8 = ( 16*Y6[::2] - Y5) / 15 # from 4th order to 6th order
            Y9 = ( 16*Y7[::2] - Y6) / 15 # from 4th order to 6th order
            
            Y10 = ( 64*Y9[::2] - Y8) / 63 # from 6th order to 8th order
            
            self.error_norm = LA.norm(Y10 - Y9[::2], np.inf, axis=0)
            y[:], dy[:], d2y[:] = Y10.T[ [0, 1, 2] ]
            Y[:] = Y10
        
        self.y = y
        self.dy = dy
        self.d2y = d2y
        sol = BPoly.from_derivatives(x, Y)
        self.sol = sol
    
    def __repr__(self):
        attrs = ['sol', 'x', 'y', 'dy', 'd2y', 'error_norm', 
                 'res_norm', 'niter', 'status', 'message', 'success']
        m = max( map(len, attrs) ) + 1
        return '\n'.join([ a.rjust(m) + ': ' + repr(getattr(self,a)) for a in attrs ])
    
    @staticmethod
    def _validate_input(f, y, a, b, wa, wb, fy=None, fyp=None, order=2, ftol=0.0, ytol=0.0, rtol=1e-10, maxiter=None):
        # CHeck y
        y = np.asfarray(y)
        if y.size < 3:
            raise ValueError('The number of data points must be at least 3')
        else:
            pass
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
        # Check fy and fyp
        if fy is None:
            fy=lambda x,y,yp: 1e5 * ( -.5*f(x,y-1e-5,yp) + .5*f(x,y+1e-5,yp) )
        if fyp is None:
            fyp=lambda x,y,yp: 1e5 * (-.5*f(x,y,yp-1e-5) + .5*f(x,y,yp+1e-5)) 
        # Check order input
        if order!=2 and order!=4 and order!=6 and order!=8:
            raise ValueError('order can only be 2, 4, 6, or 8 only.')
        else:
            pass
        # Check ftol, ytol, and rtol
        if type(ftol)!=float or type(ytol)!=float or type(rtol)!=float:
            raise ValueError('ftol, ytol, and rtol must have type float.')
        elif ftol<0 or ytol<0 or rtol<0:
            raise ValueError('ftol, ytol, and rtol must be nonnegative.')
        elif rtol<1e-12:
            warn(f'rtol value of {rtol} is too small. Setting it to {1e-12}.')
            rtol = max(rtol, 1e-12)
        else:
            pass
        # Check maxiter
        if maxiter is None:
            maxiter=100*y.size
        elif type(maxiter) is not int:
            raise ValueError('maxiter data type must be an integer.')
        elif maxiter < 1:
            raise ValueError('maxiter must be a positive integer.')
        else:
            pass
        
        return f, y, a, b, wa, wb, fy, fyp, order, ftol, ytol, rtol, maxiter
    
    def get_derivatives(self, y, N, h):
        """
        Obtain the first and second derivatives of y.
        The second derivative is obtained using the appropriate 2nd-order finte difference formula,
        not by recursive use of np.gradient on the 1st derivative since it leads to less acurate results.
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
    
    def generate_and_solve(self, N, y):
        """
        Helper function for repeatedly solving the boundary-value problem
        on finer meshes.
        Used for richardson extrapolation to achieve higher-order-accurate results.
        """
        x, h = np.linspace(self.x[0], self.x[-1], N + 1, retstep=True)
        
        guess = np.interp(x, self.x, y)
        
        ygen = self.sparse_newton(self.F, guess, self.J, self.ftol, self.ytol, self.rtol, self.maxiter, [N, h, x])
        
        dygen, d2ygen = self.get_derivatives(ygen, N, h)
        
        return np.array([ygen, dygen, d2ygen]).T
    
    def F(self, y, N, h, x):
        """ 
        Vector function F evaluating the residuals of the nonlinear system of equations. 
        It is generated based on the number of meshes, mesh size, and the grid points.
        The boundary conditions are included in the vector of residuals and they're the first and last
        elements, respectively.
        """
        w1 = self.w1
        w2 = self.w2
        w3 = self.w3
        w4 = self.w4
        w5 = self.w5
        w6 = self.w6
        f = self.f
        
        res = np.zeros(N+1)
        res[0] = y[0]*(2*h*w1 - 3*w2) + y[1]*(4*w2) + y[2]*(-w2) - 2*h*w3
        res[-1] = y[N-2]*(w5) + y[N-1]*(-4*w5) + y[N]*(2*h*w4 + 3*w5) - 2*h*w6
        fi = f( x[1:N], y[1:N], (y[2:N+1] - y[0:N-1])/(2*h) )
        res[1:-1] = -y[0:N-1] + 2*y[1:N] - y[2:N+1] + h**2*fi
        return res
    
    def J(self, y, N, h, x):
        """
        Efficiently construct the large, sparse jacobian matrix using sparse matrices to avoid large memory allocation.
        If the boundary conditions are pure dirchlet boundary conditons, the resulting jacobian matrix is a tridiagonal
        matrix. Otherwise, if one of the boundary conditions involves the derivative, either neumann or robin
        boundary conditions, the jacobian matrix is almost-symmetric and almost tridiagonal. In either case,
        the jacobian is fast and efficiently generated.
        """
        w1 = self.w1
        w2 = self.w2
        w4 = self.w4
        w5 = self.w5
        fy = self.fy
        fyp = self.fyp
        
        xi=x[1:N]
        yi=y[1:N]
        yip1=y[2:N+1]
        yim1=y[0:N-1]
        dyi=(yip1 - yim1)/(2*h)
        fyi=fy(xi, yi, dyi)
        fypi=fyp(xi, yi, dyi)
        
        p=-1 - .5*h*fypi
        q=2 + h**2*fyi
        r=-1 + .5*h*fypi
        
        k0=np.hstack((2*h*w1 - 3*w2, q, 2*h*w4 + 3*w5))
        k1=np.hstack((4*w2, r))
        k2=np.hstack((-w2, np.zeros(N-2) ))
        kn1=np.hstack((p, -4*w5))
        kn2=np.hstack(( np.zeros(N-2), w5))
        A=diags([kn2,kn1,k0,k1,k2], [-2,-1,0,1,2], shape=(N+1, N+1), format='csc', dtype=float)
        return A
    
    def sparse_newton(self, F, y, J, ftol, ytol, rtol, maxiter, args):
        """
        Sparse newton solver for solving the system of nonlinear equations for the
        discretized boundary-value problem.
        """
        F = lambda y, F=F: F(y, *args)
        J = lambda y, J=J: J(y, *args)
        self.success = False
        for i in range(1, maxiter+1):
            Ji=J(y)
            lu=splu(Ji)
            determinant=lu.L.diagonal().prod() * lu.U.diagonal().prod()
            Fi=F(y)
            if abs(determinant) <= 1e-8:
                self.niter += i-1
                self.message = f'Singular jacobian encountered in iteration {i-1}.'
                self.status = 2
                self.res_norm = LA.norm(Fi, np.inf)
                break
            dy=lu.solve(-Fi)
            y+=dy
            res=F(y)
            res_norm=LA.norm(res, np.inf)
            dy_norm=LA.norm(dy, np.inf)
            
            if res_norm <= ftol or dy_norm <= ytol + LA.norm(y,np.inf)*rtol:
                self.niter += i
                self.status = 0
                self.message = 'The algorithm converged to the desired accuracy.'
                self.res_norm = res_norm
                self.success = True
                break
        else:
            self.niter += maxiter
            self.message = f"Solution not found in {maxiter} iterations."
            self.status = 1
        
        return y