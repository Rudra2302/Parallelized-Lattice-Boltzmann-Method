# Parallelized Lattice Boltzmann Method
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
be thoroughly examined and efficient parallel solutions will be devised.

In this paper, we delve into the potential of parallel computing paradigms, specifically OpenMP, MPI, CUDA and SYCL to accelerate LBM simulations. By harnessing both OpenMP and MPI, known as hybrid parallelization, we aim to achieve optimal performance gains.

All equations, approximations and boundary condition equations can be found in the `LBM_Report.pdf` file included in the report. The report also details the parallelism stratergy and results which shall be briefly discussed here.

## Authors
Rudra Panch - IIT Madras(Main contributor)
Om Raul Deepak - IIT Madras
Rachit Kumar - IIT Madras

## Parallelsim Strategy

- The function value at any point in space at an instant of time depends on the function values of itself and all neighbouring points in the previous instant. This leads to a loop carried dependency over the time loop
- However, at an instant of time, to calculate the function value at any point, we require the neighbouring function values which leads to data dependency. But, this data having been calculated in the previous iteration is already available on the memory for all threads to access. Hence, the
inner loop for calculating values at a certain time is parallelizable.
- Inside the time loop, we have 3 more loops. One on the x direction, one on the y direction and one
for the 9 different directions of the LBM model. The 2 loops for directions can be parallelized. There is no need to parallelize the 3rd loop as it runs only over 9 iterations.


## Open-MP Implementation

To run the OpenMP code for LBM, use the following command line argument. an `#ifdef` condition has been added on the `#include<omp.h>` which makes sure a suitable version of OpenMP exists.

```
$ gcc -fopenmp LBM_openmp.c -o LBM_openmp
$ ./LBM_openmp num_threads
```

Replace num_threads with the number of the threads that you want to run the code with. Instead you could also set the environment variable `export OMP_NUM_THREADS=4` and comment the lines which access the command line through `argc` and `argv`. With the environment variable set, there is no need to mention num_threads in the command line.

## MPI Implementation

Make sure you have an MPI compiler up and running. You can check this by running the commands. If these return an empty string, please install/configure MPI properly before running the code.

```
$ which mpicc
$ which mpiexec
```

If these do not return an empty string, proceed with the next commands to run the code

```
$ mpicc LBM_mpi.c -o LBM_mpi
$ mpirun -n num_procs ./LBM_mpi
```

Replace num_procs with the processors you want to run the code on. Please check the system's specifications before setting num_procs

## Hybrid OpenMP and MPI Implementation

This code uses both OpenMP and MPI for parallelization. This code can be run using `mpicc` compiler and `-fopenmp` flag. `num_threads` and `num_procs` both have to be specified in the command line or through environment variables

```
$ mpicc -fopenmp LBM_hybrid.c -o LBM_hybrid
$ mpirun -n num_procs ./LBM_hybrid num_threads
```

## CUDA Implementation

Ensure the system has CUDA support and install nvidia-cuda-toolkit to setup the `nvcc` compiler. Make sure these are done by using

```
$ nvidia-smi
$ nvcc --version
```

To run the code, execute these commands on the terminal

```
$ nvcc LBM_cuda.cu -o LBM_cuda
$ ./LBM_cuda
```

To set number of blocks and number threads per block, check the code for 2 variables `NUM_BLOCKS` and `NUM_THREADS`. These variable can be modified but do so with caution that they do not overflow on the maximum available.

## SYCL Implementation

Ensure the system has support for SYCL and the `icpx` compiler is configured. SYCL kernels can be supported on Intel GPUs as well as NVIDIA GPUs but MKL BLAS is not supported on NVIDIA GPUs. You can check the system requirements by using the following piece of code.

```
$ sycl-ls
$ icpx --version
```

In case icpx is not configured, try running

```
source /path/to/intel/compiler/setvars.sh
```

Replace `/path/to/intel/compiler/` with the path on the system. Usually looks like `/opt/intel/compiler/`. After configuration, compile and run the code using the commands listed.

```
$ icpx -fsycl -DMKL_ILP64 LBM_sycl.cpp -o LBM_sycl
$ ./LBM_sycl
```

Check the code for changing number of workgroups and number of workitems per group

## Conclusion

Successfully parallelized the Lattice-Boltzmann method and studied 5 different parallel implementations. The results can be seen in the report attached. Testing on CUDA and SYCL is going on and the results will be updated in some time
