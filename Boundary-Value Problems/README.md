# Boundary-Value Problems
Methods for solving boundary-value problems.

I would like to express great thanks and credit to Douglas Harder (https://www.youtube.com/c/DouglasHarder/videos)
for his very helpful, clear, and enlightening lectures about the linear finite difference method for boundary-value problems:

7.1.2.2 Finite difference methods for boundary value problems
https://www.youtube.com/watch?v=Dxi0gc8-f8c&t=874s

7.1.2.3 Neumann and insulated boundary conditions
https://www.youtube.com/watch?v=OJ4SQuua-f0&t=731s
        
His lectures allowed me to grasp the derivation of the linear finite difference method for linear
boundary-value problems. I extended and generalized his algorithm to be able to deal with robin boundary conditions.

Also special thanks to Kevin Mooney (https://www.youtube.com/user/kpmooney/videos)
for his videos related to the topic, particularly the following:

Sparse Matrices to Speed up Calculations (Part 2): Partial Differential Equations - 1-D Diffusion
https://www.youtube.com/watch?v=qo-WzsVnXGE

where I became aware of SciPy's sparse matrix solvers. This allowed the algorithm to solve the system of linear equations
very fast even for thousands of meshes, although things begin to slowdown at around 10,000 meshes and greater. More importantly,
a finer mesh and ever decreasing mesh size doesn't guarantee more accurate results due to accumulating round-off error.
