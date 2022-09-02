# Python-Numerical-Analysis
Numerical Analysis algorithms/methods in Python.

# Current Implementations

**Numerical Integration**
- adaptive Newton-Cotes quadrature

**Numerical Differentiation**
- Finite Difference Coefficients Calculator
  - Lagrange polynomial method
  - Taylor series method
- Jacobian, Gradient, and Hessian approximations
- Richardson Extrapolation

**Interpolation**
- Linear Splines

**Boundary-Value Problems**
- Linear Finite Difference Method
- Nonlinear Finite Difference Method

**Initial-Value Problems**
- Scalar/systems of **1st-order** ordinary differential equations (ODEs)
  - Runge-Kutta methods
    - Fixed-step or adaptive-step using **step-doubling** (orders **1** to **8**)
    - Runge-Kutta-Fehlberg methods (orders **1** to **8**)
  - Linear Multistep methods
    - Adams-Bashforth-Moulton Predictor-Corrector (orders 1 to 5, fixed-step)
- Direct methods for solving **2nd-order** ODEs
  - Problems of the special form $\frac{d^{2}y}{dt^{2}} = f(t,y)$
    - Runge-Kutta-Nystrom methods (**work in progress**)
  - General second-order ODEs $\frac{d^{2}y}{dt^{2}} = f(t,y,\frac{dy}{dt})$
    - Runge-Kutta-Nystrom-Generalized (RKNG) methods
      - **Direct** RKNG pairs (orders **4** to **7**)
      - Transformed Nystrom pairs (**work in progress**)
    - Direct Adams-Bashforth-Moulton (DABM) methods (**work in progress**)
