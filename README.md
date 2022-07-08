# Python-Numerical-Analysis
Numerical Analysis algorithms/methods in Python.

# Current Implementations

**Numerical Differentiation**
- Finite Difference Coefficients Calculator
- Jacobian, Gradient, and Hessian approximations

**Interpolation**
- Linear Splines

**Boundary-Value Problems**
- Linear Finite Difference Method
- Nonlinear Finite Difference Method

**Initial-Value Problems**
- Scalar/systems of 1st-order differential equations
  - Runge-Kutta methods (Order 1 to 5, fixed-step or adaptive using step-doubling)
- Direct methods for solving 2nd-order initial-value problems
  - Problems of the special form $\frac{d^{2}y}{dt^{2}} = f(t,y)$
    - Explicit Central Difference Method / Verlet integration (**work in progress**)
    - Runge-Kutta-Nystrom methods (**work in progress**)
  - Problems of the more general form $\frac{d^{2}y}{dt^{2}} = f(t,y,\frac{dy}{dt})$
    - Implicit Central Difference Method / Verlet integration (**work in progress**)
    - Generalized Runge-Kutta-Nystrom methods (**work in progress**)
