# Initial-Value Problems
Methods for solving initial-value problems
- Scalar/systems of 1st-order differential equations
  - Runge-Kutta methods (Order 1 to 5, fixed-step or adaptive using step-doubling)
  - Linear Multistep methods (Order 1 to 5, fixed-step)
- Direct methods for solving 2nd-order initial-value problems
  - Problems of the special form $\frac{d^{2}y}{dt^{2}} = f(t,y)$
    - Explicit Central Difference Method / Verlet integration (**work in progress**)
    - Runge-Kutta-Nystrom methods (**work in progress**)
  - General second-order ODEs $\frac{d^{2}y}{dt^{2}} = f(t,y,\frac{dy}{dt})$
    - Implicit Central Difference Method / Verlet integration (**work in progress**)
    - Runge-Kutta-Nystrom-Generalized (RKNG) methods (**work in progress**)
