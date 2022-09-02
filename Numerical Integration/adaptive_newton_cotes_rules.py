import numpy as np
from warnings import warn

eps = np.finfo(float).eps

def _quad_ancr_mem(f, a, b, fa, fb, order):
    """
    Evaluates newton-cotes rules of specified order, also returning m and f(m) to reuse
    """
    m = (a + b) / 2
    fm = f(m)
    if order==2:
        h = abs(b - a)
        whole = .5 * h * (fa + fb)
    elif order==4:
        h = abs(b - a) / 2
        whole = h/3 * (fa + 4*fm + fb)
    elif order==6:
        x, h = np.linspace(a, b, 5, retstep=True)
        whole = 2*h/45 * np.dot(
            [7, 32, 12, 32, 7],
            [fa, f(x[1]), fm, f(x[3]), fb]
            )
    elif order==8:
        x, h = np.linspace(a, b, 7, retstep=True)
        whole = h/140 * np.dot(
            [41, 216, 27, 272, 27, 216, 41],
            [fa, *f(x[1:3]), fm, *f(x[4:-1]), fb]
            )
    elif order==10:
        x, h = np.linspace(a, b, 9, retstep=True)
        whole = 4*h/14175 * np.dot(
            [989, 5888, -928, 10496, -4540, 10496, -928, 5888, 989],
            [fa, *f(x[1:4]), fm, *f(x[5:-1]), fb]
            )
    elif order==12:
        x, h = np.linspace(a, b, 11, retstep=True)
        whole = 5*h/299376 * np.dot(
            [16067, 106300, -48525, 272400, -260550, 427368, -260550, 272400, -48525, 106300, 16067],
            [fa, *f(x[1:5]), fm, *f(x[6:-1]), fb]
            )
    else:
        pass
    return m, fm, whole

def _quad_ancr(f, a, m, b, fa, fm, fb, whole, order, rtol, atol):
    """
    Efficient recursive implementation of adaptive newton-cotes rules of specified order.
    Function values at the start, middle, end of the intervals are retained.
    """
    lm, flm, left  = _quad_ancr_mem(f, a, m, fa, fm, order)
    rm, frm, right = _quad_ancr_mem(f, m, b, fm, fb, order)
    
    fac = 2**order - 1
    S1 = whole
    S2 = left + right
    E = (S2 - S1) / fac
    whole_new = S2 + E
    
    if abs(E) < max(atol, rtol*abs(whole_new)):
        pass
    else:
        Lans = _quad_ancr(f, a, lm, m, fa, flm, fm, left, order, rtol, atol/2)
        Rans = _quad_ancr(f, m, rm, b, fm, frm, fb, right, order, rtol, atol/2)
        whole_new = Lans + Rans
    
    return whole_new

def quad_ancr(f, a, b, order=4, rtol=1e-6, atol=1e-10):
    """
    Integrate f from a to b using adaptive newton-cotes rule of specified order
    with relative and maximum tolerances given.
    
    
    References
    --------
    [1] Adaptive Simpson's method, https://en.wikipedia.org/wiki/Adaptive_Simpson%27s_method
    [2] Wolfram Mathworld, Newton-Cotes Formulas, https://mathworld.wolfram.com/Newton-CotesFormulas.html
    """
    if order not in list(range(2, 14, 2)):
        raise ValueError('order argument must be among [2,4,6,8,10,12].')
    
    if type(rtol) is not float:
        raise TypeError('rtol must be a float')
    elif rtol < 0:
        raise ValueError('rtol values must be positive.')
    elif rtol < 1000*eps:
        rtol = 1e3*eps
        warn(f'rtol value too small, setting to {rtol}')
    else:
        pass
    
    if type(atol) is not float:
        raise TypeError('atol must be a float')
    elif atol < 0:
        raise ValueError('atol values must be nonnegative.')
    else:
        pass
    
    fa, fb = f(a), f(b)
    m, fm, whole = _quad_ancr_mem(f, a, b, fa, fb, order)
    return _quad_ancr(f, a, m, b, fa, fm, fb, whole, order, rtol, atol)