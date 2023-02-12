import numpy as np
from mpmath import fp

eps = np.finfo(float).eps

def numderivative(f, x, h=None, method='central', order=2, hessdiag=False, H_form='default', args=None):
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
        
    method : {'central', 'complex', 'forward', 'backward'}, optional
        The finite difference scheme to use.
    
    order : {2, 4, 6, 8, 10}, optional
        The order of the central finite difference method to use. The mixed second-order
        partial derivatives of the Hessian are not approximated if the input vector x has only
        one component. Otherwise, they are efficiently approximated with minimum additional
        function evaluations. For instance, the efficient second-order approximation for the
        mixed partial derivatives can be found in wikipedia [1]. It only requires two (2) more
        function evaluation for each unique, off-diagonal entry. Taylor series expansion shows
        that the 2nd order Hessian approximation is exact for polynomials of degree three (3)
        or less, since its truncation error is proportional to the fourth derivatives.
        
        The other methods also require minimum additional function evaluations.
        They are derived using 2-d taylor series expansion in a Jupyter Lab notebook with the help of SymPy.
        
    hessdiag : {True, False}, optional
        Specify whether to compute only the diagonal of the hessian.
        True by default.
        
    H_form : string, {'default', 'blockmat', 'hypermat'}, optional
        The format of the 3-d hessian.
        
        'default' : the 3-d array has shape (m,n,n), where m and n are the lengths
        of the output and input vectors, respectively.
        
        'blockmat' : The hessian matrices are stacked row-by-row resulting in a 2-d array
        shape of (m*n,n).
        
        'hypermat' : The output is a (n,n,m) 3-d array where H[:,:,k] corresponds to the kth
        hessian matrix.
        
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
    f, x, h, fx, n, m = validate_inputs(f, x, h, method, order, hessdiag, H_form, args)
    
    stencil, stencil_weights = csten(method,order)
    
    # Allocate 3d array and use 3-d indexing if the full Hessian is needed.
    # Otherwise, the Hessian diagonal array has the same (m,n) shape as the Jacobian.
    if method=='complex' and hessdiag is False:
        e_vec = np.eye(n)
        J = np.zeros((m,n))
        H = np.zeros((m,n,n))
        for i in range(n):
            J[:,i] = np.imag( f(x + 2.22e-16j*e_vec[i]) ) / 2.22e-16
            f_vals = np.array([
                fx if si==0 else f(x + si * h[i] * 1j) for si in stencil
                ]).real
            H[:,i] = stencil_weights @ f_vals / h[i,i]**2
        
        # Compute off-diagonal elements of Hessian
        # Exploit the symmetry of the Hessian
        if n>1:
            for i in range(n-1):
                for j in range(i+1,n):
                    r = h[i,i]/h[j,j]
                    f_vals = np.array([
                        fx if si==0 else f(x + si * (h[i]+h[j]) * 1j) for si in stencil
                        ]).real
                    H[:,i,j] = stencil_weights @ f_vals / (2*h[i,i]*h[j,j]) - (r**2*H[:,i,i] + H[:,j,j]) / (2*r)
                    H[:,j,i] = H[:,i,j]
        
        if m>1 and H_form=='default':
            pass
        elif m>1 and H_form=='hypermat':
            H = H.T
        elif m>1 and H_form=='blockmat':
            H = H.reshape((m*n,n))
        
    elif method=='complex' and hessdiag is True:
        e_vec = np.eye(n)
        J = np.zeros((m,n))
        H = np.zeros((m,n))
        for i in range(n):
            J[:,i] = np.imag( f(x + 2.22e-16j*e_vec[i]) ) / 2.22e-16
            f_vals = np.array([
                fx if si==0 else f(x + si * h[i] * 1j) for si in stencil
                ]).real
            H[:,i] = stencil_weights @ f_vals / h[i,i]**2
        
    elif hessdiag is False:
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
        
        if m>1 and H_form=='default':
            pass
        elif m>1 and H_form=='hypermat':
            H = H.T
        elif m>1 and H_form=='blockmat':
            H = H.reshape((m*n,n))
        
    else:
        J = np.zeros((m,n))
        H = np.zeros((m,n))
        for i in range(n):
            f_vals = np.array([
                fx if si==0 else f(x+si*h[i]) for si in stencil
                ])
            J[:,i] = stencil_weights[0] @ f_vals / h[i,i]
            H[:,i] = stencil_weights[1] @ f_vals / h[i,i]**2
        
    if m==1:
        J = J[0]
        H = H[0]
    
    return J, H

def csten(method,order):
    idx = order//2-1
    if method=='central':
        stencil, stencil_weights = [
            [
                [1, 0, -1],
                [
                    [0.5, 0, -0.5],
                    [1, -2, 1]
                    ]
                ],
            [
                [2, np.sqrt(3), 0, -np.sqrt(3), -2],
                [
                    [-3/4, 2*np.sqrt(3)/3, 0, -2*np.sqrt(3)/3, 3/4],
                    [-3/4, 4/3, -7/6, 4/3, -3/4]
                    ]
                ],
            [
                [3.0, 2.8531695488854607163, 1.7633557568774193875, 0.0, -1.7633557568774193875, -2.8531695488854607163, -3.0],
                [
                    [0.83333333333333333333, -1.1342010778027199096, 0.70097481615884480803, 0.0, -0.70097481615884480803, 1.1342010778027199096, -0.83333333333333333333],
                    [0.55555555555555555556, -0.79504639199992522539, 0.79504639199992522539, -1.1111111111111111111, 0.79504639199992522539, -0.79504639199992522539, 0.55555555555555555556]
                    ]
                ],
            [
                [4.0, 3.8997116487272944281, 3.1273259298721192348, 1.7355349564702324819, 0.0, -1.7355349564702324819, -3.1273259298721192348, -3.8997116487272944281, -4.0],
                [
                    [-0.875, 1.1523824354812432526, -0.51285843163627694975, 0.63952400384496630287, 0.0, -0.63952400384496630287, 0.51285843163627694975, -1.1523824354812432526, 0.875],
                    [-0.4375, 0.59100904850610352546, -0.3279852776056817678, 0.73697622909957824234, -1.125, 0.73697622909957824234, -0.3279852776056817678, 0.59100904850610352546, -0.4375]
                    ]
                ],
            [
                [5.0, 4.9240387650610402968, 4.3301270189221932338, 3.2139380484326966316, 1.7101007166283436652, 0.0, -1.7101007166283436652, -3.2139380484326966316, -4.3301270189221932338, -4.9240387650610402968, -5.0],
                [
                    [0.9, -1.1695217600652349009, 0.46188021535170061161, -0.40617064475429799409, 0.62228953074416492802, 0.0, -0.62228953074416492802, 0.40617064475429799409, -0.46188021535170061161, 1.1695217600652349009, -0.9],
                    [0.36, -0.47502540733987785937, 0.21333333333333333333, -0.25275573992620701284, 0.72778114726608487222, -1.1466666666666666667, 0.72778114726608487222, -0.25275573992620701284, 0.21333333333333333333, -0.47502540733987785937, 0.36]
                    ]
                ]
            ][idx]
    elif method=='complex':
        # Only the stencil for the second derivative is stored since the 1st cs-step approximation
        # for the 1st derivative doesn't suffer from subtractive cancellation.
        stencil, stencil_weights = [
            [
                [1,0],
                [-2, 2]
                ],
            [
                [2,1,0],
                [1/6, -8/3, 5/2]
                ],
            [
                [3,2,1,0],
                [-1/45, 3/10, -3, 49/18]
                ],
            [
                [4,3,2,1,0],
                [1/280, -16/315, 2/5, -16/5, 205/72]
                ],
            [
                [5,4,3,2,1,0],
                [-1/1575, 5/504, -5/63, 10/21, -10/3, 5269/1800]
                ]
            ][order//2-1]
    elif method=='forward':
        stencil, stencil_weights = [
            [
                [2.0, 1.0, 0.0],
                [
                    [-0.5, 2.0, -1.5],
                    [1.0, -2.0, 1.0]
                    ]
                ],
            [
                [4.0, 3.7320508075688772935, 2.0, 0.26794919243112270647, 0.0],
                [
                    [-0.25, 0.35726558990816360863, -0.33333333333333333333, 4.9760677434251697247, -4.75],
                    [2.25, -3.2025650515289120796, 2.8333333333333333333, -10.130768281804421254, 8.25]
                    ]
                ],
            [
                [6.0, 5.8531695488854607163, 4.7633557568774193875, 3.0, 1.2366442431225806125, 0.14683045111453928365, 0.0],
                [
                    [-0.16666666666666666667, 0.22114978563134156984, -0.10379808190602873566, 0.13333333333333333333, -0.39981360342685595001, 8.8157952330348764492, -8.5],
                    [2.7777777777777777778, -3.6839805332209841453, 1.7209854778011433586, -2.1777777777777777778, 6.1502207141969125016, -29.787225658777071715, 25.0]
                    ]
                ],
            [
                [8.0, 7.8997116487272944281, 7.1273259298721192348, 5.7355349564702324819, 4.0, 2.2644650435297675181, 0.87267407012788076517, 0.10028835127270557193, 0.0],
                [
                    [-0.125, 0.16253610284475008058, -0.064294819579528325188, 0.055290210072491050444, -0.071428571428571428571, 0.14004143430142832346, -0.52511029080819412082, 12.80296593459762442, -12.375],
                    [3.0625, -3.9816186629632006111, 1.5732550056113784883, -1.3491528202620787656, 1.7321428571428571429, -3.3423393906659728158, 11.793028583775442347, -61.550315572638425785, 52.0625]
                    ]
                ],
            [
                [10.0, 9.9240387650610402968, 9.3301270189221932338, 8.2139380484326966316, 6.7101007166283436652, 5.0, 3.2898992833716563348, 1.7860619515673033684, 0.66987298107780676618, 0.075961234938959703166, 0.0],
                [
                    [-0.1, 0.12895221434819206716, -0.047635411987755147817, 0.035316861051302741669, -0.035242980608779715258, 0.044444444444444444444, -0.07188182040537447448, 0.16241906305990473172, -0.66347569912335596329, 16.847103329221421316, -16.3],
                    [3.24, -4.1778543379651080565, 1.5427033346622235681, -1.1427304184049582618, 1.1384167113269763583, -1.4311111111111111111, 2.2996488522134972965, -5.1129875146867639203, 19.648407776448887543, -105.64449329248364342, 89.64]
                    ]
                ]
            ][idx]
    else:
        stencil, stencil_weights = [
            [
                [0.0, -1.0, -2.0],
                [
                    [1.5, -2.0, 0.5],
                    [1.0, -2.0, 1.0]
                    ]
                ],
            [
                [0.0, -0.26794919243112270647, -2.0, -3.7320508075688772935, -4.0],
                [
                    [4.75, -4.9760677434251697247, 0.33333333333333333333, -0.35726558990816360863, 0.25],
                    [8.25, -10.130768281804421254, 2.8333333333333333333, -3.2025650515289120796, 2.25]
                 ]
                ],
            [
                [0.0, -0.14683045111453928365, -1.2366442431225806125, -3.0, -4.7633557568774193875, -5.8531695488854607163, -6.0],
                [
                    [8.5, -8.8157952330348764492, 0.39981360342685595001, -0.13333333333333333333, 0.10379808190602873566, -0.22114978563134156984, 0.16666666666666666667],
                    [25.0, -29.787225658777071715, 6.1502207141969125016, -2.1777777777777777778, 1.7209854778011433586, -3.6839805332209841453, 2.7777777777777777778]
                 ]
                ],
            [
                [0.0, -0.10028835127270557193, -0.87267407012788076517, -2.2644650435297675181, -4.0, -5.7355349564702324819, -7.1273259298721192348, -7.8997116487272944281, -8.0],
                [
                    [12.375, -12.80296593459762442, 0.52511029080819412082, -0.14004143430142832346, 0.071428571428571428571, -0.055290210072491050444, 0.064294819579528325188, -0.16253610284475008058, 0.125],
                    [52.0625, -61.550315572638425785, 11.793028583775442347, -3.3423393906659728158, 1.7321428571428571429, -1.3491528202620787656, 1.5732550056113784883, -3.9816186629632006111, 3.0625]
                    ]
                ],
            [
                [0.0, -0.075961234938959703166, -0.66987298107780676618, -1.7860619515673033684, -3.2898992833716563348, -5.0, -6.7101007166283436652, -8.2139380484326966316, -9.3301270189221932338, -9.9240387650610402968, -10.0],
                [
                    [16.3, -16.847103329221421316, 0.66347569912335596329, -0.16241906305990473172, 0.07188182040537447448, -0.044444444444444444444, 0.035242980608779715258, -0.035316861051302741669, 0.047635411987755147817, -0.12895221434819206716, 0.1],
                    [89.64, -105.64449329248364342, 19.648407776448887543, -5.1129875146867639203, 2.2996488522134972965, -1.4311111111111111111, 1.1384167113269763583, -1.1427304184049582618, 1.5427033346622235681, -4.1778543379651080565, 3.24]
                    ]
                ]
            ][idx]
    return stencil, stencil_weights

def validate_inputs(f, x, h, method, order, hessdiag, H_form, args):
    if args is None:
        f = lambda u, f=f: np.atleast_1d( f(u) )
    else:
        args_type = (isinstance(args,list) or  isinstance(args,tuple))
        if not args_type:
            raise TypeError('Argument args must be a tuple or list.')
        elif args_type and len(args)==0:
            raise ValueError('Argument args must be a non-empty tuple or list.')
        else:
            f = lambda u, f=f: np.atleast_1d( f(u,*args) )
    
    if type(hessdiag) is not bool:
        raise TypeError('Argument hessdiag must be a boolean.')
    else:
        pass
    
    if type(H_form) is not str:
        raise TypeError('Argument H_form must be a string.')
    elif H_form not in ['default', 'blockmat', 'hypermat']:
        raise ValueError(f"Value of H_form must be among {['default', 'blockmat', 'hypermat']}")
    else:
        pass
    
    if type(method) is not str:
        raise TypeError('Argument method must be a string.')
    elif method not in ['central', 'complex', 'forward', 'backward']:
        raise ValueError(f"Value of method must be among {['central', 'complex', 'forward', 'backward']}")
    else:
        pass
    
    x = np.atleast_1d(x)
    if x.ndim>1:
        raise Exception('Input x should be at most 1-d array-like.')
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
    elif fx.dtype not in [np.int32, np.int64, np.float32, np.float64]:
        raise TypeError('Output of f must be real numbers of type int or float.')
    else:
        pass
    
    n = x.size
    m = fx.size
    
    if h is None:
        cons = {'central': 2, 'complex': 2, 'forward': 1, 'backward': 1}[method]
        h_def = fp.nthroot(eps,order + cons)
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
    
    order_list = [2, 4, 6, 8, 10]
    if type(order) is not int:
        raise TypeError('Argument order must be an integer.')
    elif order % 2 != 0:
        raise ValueError('Argument order must be an even integer.')
    elif order not in order_list:
        raise Exception(f"Value for order must be in {order_list}")
    else:
        pass
    
    return f, x, h, fx, n, m
