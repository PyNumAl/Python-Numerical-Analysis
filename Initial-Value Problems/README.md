# Initial-Value Problems
Methods for solving initial-value problems
- Scalar/systems of 1st-order differential equations
  - Runge-Kutta methods
    - 1st-to-5th order RK (fixed-step or adaptive using step-doubling)
    - Runge-Kutta-Fehlberg (**RKF45** and **RKF78**)
  - Linear Multistep methods
    - Adams-Bashforth-Moulton Predictor-Corrector (Order 1 to 5, fixed-step)
- Direct methods for solving 2nd-order initial-value problems
  - Problems of the special form $\frac{d^{2}y}{dt^{2}} = f(t,y)$
    - Runge-Kutta-Nystrom methods (**work in progress**)
  - General second-order ODEs $\frac{d^{2}y}{dt^{2}} = f(t,y,\frac{dy}{dt})$
    - Runge-Kutta-Nystrom-Generalized (RKNG) methods (**work in progress**)
    - Direct Adams-Bashforth-Moulton (DABM) methods (**work in progress**)
