#include <stdio.h> 
#include <string.h> 
#include <stddef.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define pi 3.14159

// Serial code for LBM

// To find the relaxation time
double find_tau(double alpha, double delx, double delt){
    double v=delx/delt;
    double t=(3*alpha)/(v*v)+(delt/2);
    return t;
}

__global__ void initialize(double *w, double *c){
    // Initialize directional weights
    for(int k=0; k<9; k++){
        if(k==4){
            w[k]=0;
        }
        else if(k%2==0){
            w[k]=(double)1/36;
        }
        else{
            w[k]=(double)2/9;
        }
    }

    // Initialize directions
    for(int i=0; i<=2; i++){
        for(int j=0; j<=2; j++){
            int k=i*3+j;
            c[k]=i-1; c[k + 9]=j-1;
        }
    }
}

__global__ void initialize(double *T, double *w, double *f, double *feq, double T_i){
    int i = blockIdx.x, j = threadIdx.x;
    int global_index = (i*blockDim.x + j)*9;

    T[i*blockDim.x + j]=T_i; //Dirichlet
    //Neumann and Robin
    // if(i==0){ 
    //     T[i*blockDim.x + j]=T_b;
    // }
    // else{
    //     T[i*blockDim.x + j]=0;
    // }

    // Initialize f and feq values 
    for(int k=0; k<9; k++){
        feq[global_index + k]=w[k]*T[i*blockDim.x + j];
        f[global_index + k]=feq[global_index + k];
    }
}

__global__ void calculate(double *c, double *f, double *feq, double *f_temp, double delt, double tau){
    int i = blockIdx.x, j = threadIdx.x;
    int global_index = (i*blockDim.x + j)*9;
    for(int k = 0; k < 9; k++){
        int ii=i+c[k], jj=j+c[k + 9];
        if(ii<0 || ii>(blockDim.x - 1) || jj<0 || jj>(blockDim.x - 1)){
            continue;
        }

        int update_index = (ii*blockDim.x + jj)*9;
        f_temp[update_index + k]=f[global_index + k]-(delt/tau)*(f[global_index + k]-feq[global_index + k]); // Calculate function value
    }
}

__global__ void calculate_boundary(double *f_temp, double *w, int n, double T_a){
    int i = threadIdx.x;
    double f=w[5]*T_a;
    f_temp[5 + i*blockDim.x*9]=f;
    f_temp[3 + (i*blockDim.x + (blockDim.x -1))*9]=f;
    if(i>0){
        f=w[2]*T_a;
        f_temp[2 + (i-1)*blockDim.x*9]=f;
        f_temp[((i-1)*blockDim.x + (blockDim.x -1))*9]=f;
    }
    if(i<n){
        f=w[8]*T_a;
        f_temp[8 + (i+1)*blockDim.x*9]=f;
        f_temp[6 + ((i+1)*blockDim.x+(blockDim.x -1))*9]=f;
    }
}

__global__ void update(double *T, double *f, double *f_temp, double *feq, double *w, double T_i){
    int i = blockIdx.x, j = threadIdx.x;
    int global_index = (i*blockDim.x + j)*9;
    T[i*blockDim.x + j]=0;
    for(int k=0; k<9; k++){
        if(i==0 || i==(blockDim.x-1)){
            break;
        }
        // error+=pow(f[k + global_index]-f_temp[k + global_index],2); // Calculate error between successive iterations
        f[k + global_index]=f_temp[k + global_index];
        T[i*blockDim.x + j]+=f[k + global_index]; // Calculate new temperature value
    }

    // Boundary conditions
    if(i==0){
        T[i*blockDim.x + j]=T_i; // Dirichlet
        // T[i*blockDim.x + j]=T_b; // Neumann and Robin
    }
    if(i==blockDim.x-1){
        T[i*blockDim.x + j]=T_l; //Dirichlet 
        // T[i*blockDim.x + j]=T[(i-1)*blockDim.x + j]-((q*delx)/lambda); //Neumann
        // T[i*blockDim.x + j]=T[(i-1)*blockDim.x + j]*(lambda/(lambda+alpha*delx))+T_a*((alpha*delx)/(lambda+alpha*delx)); //Robin
    }

    // Calculate new equilibrium values
    for(int k=0; k<9; k++){
        feq[k + global_index]=w[k]*T[i*blockDim.x + j];
    }
}

int main(){

    // Declare variables
    clock_t start=clock();
    double delx=0.0001, dely=0.0001, delt=0.1;
    double T_i=0, T_l=100, T_a=25, T_b=150, alpha=10, tau=0.1, q=1000, lambda=35;
    int n=100, size = (n+1)*(n+1)*sizeof(double); // grid size

    const int NUM_BLOCKS = n+1;
    const int NUM_THREADS = n+1;

    cudaMalloc((void**)&f, size*9);
    cudaMalloc((void**)&feq, size*9);
    cudaMalloc((void**)&T_device, size);
    cudaMalloc((void**)&w, 9);
    cudaMalloc((void**)&c, 18);

    // Initialize directions and directional weights
    initialize<<< 1,1 >>>(w,c);

    // Initialize temperature values
    initialize<<< NUM_BLOCKS, NUM_THREADS >>>(T_device,w,f,feq,T_i);

    cublasHandle_t handle;
    cublasCreate(handle);

    // Run for the time required
    for(int it=1; it<=50000; it++){
        // Declare a temperary function matrix
        cudaMalloc((void**)&f_temp, size*9);
        cudaMalloc((void**)&device_error, sizeof(double));
        cudaMemset(device_error, 0, sizeof(double));

        // calculate new f values
        calculate<<< NUM_BLOCKS, NUM_THREADS >>>(c,f,feq,f_temp,delt,tau);

        // Impact on points just outside boundary on the boundary points of j=0 and j=n-1
        calculate_boundary<<< 1, NUM_THREADS >>>(f_temp,w,n,T_a);

        cublasStatus_t status = cublasDnrm2(handle, size, f-f_temp, 1, device_error);

        // Update values
        update<<< NUM_BLOCKS, NUM_THREADS >>>(T_device,f,f_temp,feq,w,T_i);

        // Set an error threshold for breaking and print error values
        double host_error = 0;
        cudaMemcpy(&host_error, device_error, sizeof(double), cudaMemcpyDeviceToHost);
        // host_error=sqrt(host_error);
        printf("%d %.10f\n",it, host_error);
        if(host_error<0.000001 && it!=1){
            break;
        }

        cudaFree(f_temp);
        cudaFree(device_error);
    }

    cublasDestroy(handle);

    double T[(n+1)*(n+1)];
    cudaMemcpy(T, T_device, size, cudaMemcpyDeviceToHost);

    // Print temperature values of whole grid
    for(int i=0; i<(n+1)*(n+1); i++){
        printf("%f ",T[i]);
        if(i%(n+1) == n)
            printf(";");
    }
    // Temperature values on the midplane
    for(int i=(n/2); i<(n+1)*(n+1); i+=(n+1)){
        printf("%f ",T[i]);
    }
    // Print time of execution
    clock_t end=clock();
    printf("\nTime taken: %ld\n",(end-start));
}