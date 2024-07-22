#include <stdio.h> 
#include <string.h> 
#include <stddef.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#define pi 3.14159
using namespace std;

// Serial code for LBM

// To find the relaxation time
double find_tau(double alpha, double delx, double delt){
    double v=delx/delt;
    double t=(3*alpha)/(v*v)+(delt/2);
    return t;
}

void initialize(double *w, double *c){
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

void initialize(sycl::nd_item<1> ind, double *T, double *w, double *f, double *feq, double T_i){
    // int i = ind.get_group(0), j = ind.get_local_id(0);
    int global_index = ind.get_global_id(0)*9;

    T[ind.get_global_id(0)]=T_i; //Dirichlet
    //Neumann and Robin
    // if(i==0){ 
    //     T[ind.get_global_id(0)]=T_b;
    // }
    // else{
    //     T[ind.get_global_id(0)]=0;
    // }

    // Initialize f and feq values 
    for(int k=0; k<9; k++){
        feq[global_index + k]=w[k]*T[ind.get_global_id(0)];
        f[global_index + k]=feq[global_index + k];
    }
}

void calculate(sycl::nd_item<1> ind, double *c, double *f, double *feq, double *f_temp, double delt, double tau, int n){
    int i = ind.get_group(0), j = ind.get_local_id(0);
    int global_index = (ind.get_global_id(0))*9;
    for(int k = 0; k < 9; k++){
        int ii=i+c[k], jj=j+c[k + 9];
        if(ii<0 || ii>n || jj<0 || jj>n){
            continue;
        }

        int update_index = (ii*(n+1) + jj)*9;
        f_temp[update_index + k]=f[global_index + k]-(delt/tau)*(f[global_index + k]-feq[global_index + k]); // Calculate function value
    }
}

void calculate_boundary(sycl::nd_item<1> ind, double *f_temp, double *w, double T_a, int n){
    int i = ind.get_global_id(0);
    double f=w[5]*T_a;
    f_temp[5 + i*(n+1)*9]=f;
    f_temp[3 + (i*(n+1) + n)*9]=f;
    if(i>0){
        f=w[2]*T_a;
        f_temp[2 + (i-1)*(n+1)*9]=f;
        f_temp[((i-1)*(n+1) + n)*9]=f;
    }
    if(i<n){
        f=w[8]*T_a;
        f_temp[8 + (i+1)*(n+1)*9]=f;
        f_temp[6 + ((i+1)*(n+1)+n)*9]=f;
    }
}

void update(sycl::nd_item<1> ind, double *T, double *f, double *f_temp, double *feq, double *w, double T_i, double T_l, int n){
    int i = ind.get_group(0), j = ind.get_local_id(0);
    int global_index = (ind.get_global_id(0))*9;
    T[ind.get_global_id(0)]=0;
    for(int k=0; k<9; k++){
        if(i==0 || i==n){
            break;
        }
        // error+=pow(f[k + global_index]-f_temp[k + global_index],2); // Calculate error between successive iterations
        f[k + global_index]=f_temp[k + global_index];
        T[ind.get_global_id(0)]+=f[k + global_index]; // Calculate new temperature value
    }

    // Boundary conditions
    if(i==0){
        T[ind.get_global_id(0)]=T_i; // Dirichlet
        // T[ind.get_global_id(0)]=T_b; // Neumann and Robin
    }
    if(i==n-1){
        T[ind.get_global_id(0)]=T_l; //Dirichlet 
        // T[ind.get_global_id(0)]=T[(i-1)*n + j]-((q*delx)/lambda); //Neumann
        // T[ind.get_global_id(0)]=T[(i-1)*n + j]*(lambda/(lambda+alpha*delx))+T_a*((alpha*delx)/(lambda+alpha*delx)); //Robin
    }

    // Calculate new equilibrium values
    for(int k=0; k<9; k++){
        feq[k + global_index]=w[k]*T[ind.get_global_id(0)];
    }
}

int main(){

    // Declare variables
    clock_t start=clock();
    double delx=0.0001, dely=0.0001, delt=0.1;
    double T_i=0, T_l=100, T_a=25, T_b=150, alpha=10, tau=0.1, q=1000, lambda=35;
    const int n=10, size = (n+1)*(n+1); // grid size

    sycl::queue queue{sycl::gpu_selector_v};
    double* f         = sycl::malloc_device<double>(size*9, queue);
    double* feq       = sycl::malloc_device<double>(size*9, queue);
    double* T_device  = sycl::malloc_device<double>(size, queue);
    double* w         = sycl::malloc_device<double>(9, queue);
    double* c         = sycl::malloc_device<double>(18, queue);

    // Initialize directions and directional weights
    queue.parallel_for(sycl::nd_range<1>(1,1), [=] (sycl::nd_item<1> ind){
        initialize(w,c);
    });
    queue.wait();

    // Initialize temperature values
    queue.parallel_for(sycl::nd_range<1>((n+1)*(n+1),n+1), [=] (sycl::nd_item<1> ind){
        initialize(ind,T_device,w,f,feq,T_i);
    });
    queue.wait();

    // Run for the time required
    for(int it=1; it<=50000; it++){
        // Declare a temperary function matrix
        double* f_temp        = sycl::malloc_device<double>(size*9, queue);
        double* device_error  = sycl::malloc_device<double>(1, queue);
        queue.memset(device_error,0,sizeof(double));

        // calculate new f values
        queue.parallel_for(sycl::nd_range<1>((n+1)*(n+1),n+1), [=] (sycl::nd_item<1> ind){
            calculate(ind,c,f,feq,f_temp,delt,tau,n);
        });
        queue.wait();

        // Impact on points just outside boundary on the boundary points of j=0 and j=n-1
        queue.parallel_for(sycl::nd_range<1>(n+1,1), [=] (sycl::nd_item<1> ind){
            calculate_boundary(ind,f_temp,w,T_a,n);
        });
        queue.wait();

        double* f_diff = sycl::malloc_device<double>(size*9, queue);
        queue.parallel_for(sycl::nd_range<1>((n+1)*(n+1)*9,n+1), [=] (sycl::nd_item<1> ind){
            int global_index = (ind.get_global_id(0));
            if(global_index < (n+1)*9 || global_index >= n*(n+1)*9){
                f_diff[global_index] = 0;
            }
            else{
                f_diff[global_index] = f[global_index] - f_temp[global_index];
            }
        });
        queue.wait();

        oneapi::mkl::blas::row_major::nrm2(queue, size, f_diff, 1, device_error);
        queue.wait();

        // Update values
        queue.parallel_for(sycl::nd_range<1>((n+1)*(n+1),n+1), [=] (sycl::nd_item<1> ind){
            update(ind,T_device,f,f_temp,feq,w,T_i, T_l,n);
        });
        queue.wait();

        // Set an error threshold for breaking and print error values
        double host_error = 0;
        queue.memcpy(&host_error,device_error,sizeof(double));
        host_error=sqrt(host_error);
        cout<<it<<" "<<host_error<<endl;
        if(host_error<0.000001 && it!=1){
            break;
        }

        sycl::free(f_temp,queue);
        sycl::free(device_error,queue);
    }

    double T[(n+1)*(n+1)];
    queue.memcpy(T, T_device, size*sizeof(double));

    // Print temperature values of whole grid
    for(int i=0; i<(n+1)*(n+1); i++){
        cout<<T[i]<<" ";
        if(i%(n+1) == n)
            cout<<";";
    }
    cout<<endl;
    // Temperature values on the midplane
    for(int i=(n/2); i<(n+1)*(n+1); i+=(n+1)){
        cout<<T[i]<<" ";
    }
    cout<<endl;
    // Print time of execution
    clock_t end=clock();
    cout<<"Time taken: "<<(end-start)<<endl;
}