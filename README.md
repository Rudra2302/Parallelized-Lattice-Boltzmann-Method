# Parallelized-Lattice-Boltzmann-Method
The investigation detailed in this paper centers on the utilization of the Lattice Boltzmann method
(LBM) to address transient heat conduction problems, The LBM method has been widely used in
the engineering industry, especially in the study of heat transfer problems like steady and unsteady
heat flows and Fourier equations. Given its computational intensity, there exists ample opportunity
for enhancement through parallelization strategies. This study focuses primarily on solving Fourier
equations, with an emphasis on using OpenMP, MPI, CUDA and SYCL for parallelization, demonstrated through
proficient C programs.

At the heart of LBM lies the application of the Boltzmann equation and the Bhatnagar-Gross-
Krook approximation. The objective is solving the 2D transient heat conduction problem under
various boundary conditions, including Dirichlet, Neumann, or Robin. Each boundary condition will
be thoroughly examined and efficient parallel solutions will be devised
