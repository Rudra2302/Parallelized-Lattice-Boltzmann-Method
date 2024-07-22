#include <stdio.h> 
#include <string.h> 
#include <stddef.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>
#ifdef _OPENMP
#include<omp.h>
#endif
#include <mpi.h>
#define pi 3.14159
// Hybrid OpenMP+MPI code for LBM

double ab(double x){
    if(x<0){
        return -x;
    }
    return x;
}

// To find the relaxation time
double find_tau(double alpha, double delx, double delt){
    double v=delx/delt;
    double t=(3*alpha)/(v*v)+(delt/2);
    return t;
}

int main(int argc, char** argv){

    // Take number of threads from terminal
    double start=omp_get_wtime();
    int thread_count = 1;
    if (argc == 2)
        {
        thread_count = strtol(argv[1], NULL, 10);
        }
    else
        {
        printf("\n A command line argument other than name of the executable is required...exiting the program..");
        return 1;
        }


    // Declare global values
    MPI_Status status;
    int size, myid, tag=100;
    double delx=0.0001, dely=0.0001, delt=0.1;
    double T_i=0, T_l=100, T_a=25, T_b=150, alpha=10, tau=0.1, q=1000, lambda=35;
    int n=101;

    // Take number of processors from terminal
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);

    // Declare local values
    int nrows=n/size;
    int i, j, k;
    if(myid==size-1){
        nrows+=(n-nrows*size);
    }

    double f[9][nrows][n], feq[9][nrows][n], T[nrows][n]; // f for function value, feq for equilibrium value, T for temperature
    double ghost_up[3][n], ghost_down[3][n], ghost_up_eq[3][n], ghost_down_eq[3][n]; // ghost points for function value and equilibirum values
    double w[9], c[2][9]; // c for direction, w for directional weight

    // Initialize directional weights
    for(k=0; k<9; k++){
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
    w[4]=0;

    // Initialize directions
    for(i=0; i<=2; i++){
        for(j=0; j<=2; j++){
            k=i*3+j;
            c[0][k]=i-1; c[1][k]=j-1;
        }
    }

    // Initialize temperature values
    #pragma omp parallel for num_threads(thread_count) default(none) private(i,j,k) shared(myid,T,T_i,T_b,w,feq,f,n,nrows)
    for(i=0; i<nrows; i++){
        for(j=0; j<n; j++){
            //Dirichlet
            T[i][j]=T_i; 

            //Neumann and Robin
            // if(i==0 && myid==0){ 
            //     T[i][j]=T_b;
            // }
            // else{
            //     T[i][j]=0;
            // }

            // Initialize f and feq values
            for(k=0; k<9; k++){
                feq[k][i][j]=w[k]*T[i][j];
                f[k][i][j]=feq[k][i][j];
            }
        }
    }

    // Initialize ghost values
    #pragma omp parallel for num_threads(thread_count) default(none) private(j,k) shared(n,T_i,ghost_up_eq,w,ghost_up,ghost_down_eq,ghost_down)
    for(j=0; j<n; j++){
        int temp_T=T_i;
        for(k=0; k<3; k++){
            ghost_up_eq[k][j]=w[k]*temp_T;
            ghost_up[k][j]=ghost_up_eq[k][j];
        }
        for(k=6; k<9; k++){
            ghost_down_eq[k-6][j]=w[k]*temp_T;
            ghost_down[k-6][j]=ghost_down_eq[k-6][j];
        }
    }

    // Run for the time required
    for(int it=1; it<=50000; it++){
        // Declare a temperary function matrix
        double f_temp[9][nrows][n], local_error=0, global_error=0;

        // calculate new f values
        #pragma omp parallel for num_threads(thread_count) default(none) private(i,j,k) shared(nrows,n,c,f_temp,f,delt,tau,feq)
        for(i=0; i<nrows; i++){
            for(j=0; j<n; j++){
                // Run loop for the 9 directions
                for(k=0; k<9; k++){
                    int ii=i+c[0][k], jj=j+c[1][k];
                    if(ii<0 || ii>=nrows || jj<0 || jj>=n){
                        continue;
                    }
                    f_temp[k][ii][jj]=f[k][i][j]-(delt/tau)*(f[k][i][j]-feq[k][i][j]); // Calculate function value
                }
            }
        }

        // calculate the effect of ghost values on boundary points
        #pragma omp parallel for num_threads(thread_count) default(none) private(i,k) shared(n,nrows,f_temp,ghost_down,delt,tau,ghost_down_eq,ghost_up,ghost_up_eq)
        for(i=0; i<n; i++){
            for(k=6; k<9; k++){
                int jj=i+k-7;
                if(jj<0 || jj>=n){
                    continue;
                }
                f_temp[k][0][jj]=ghost_down[k-6][i]-(delt/tau)*(ghost_down[k-6][i]-ghost_down_eq[k-6][i]);
            }
            for(int k=0; k<3; k++){
                int jj=i+k-1;
                if(jj<0 || jj>=n){
                    continue;
                }
                f_temp[k][nrows-1][jj]=ghost_up[k][i]-(delt/tau)*(ghost_up[k][i]-ghost_up_eq[k][i]);
            }
        }

        // Impact on points just outside boundary on the boundary points of j=0 and j=n-1
        #pragma omp parallel for num_threads(thread_count) default(none) private(i) shared(T_a,w,f_temp,nrows,n)
        for(i=0; i<nrows; i++){
            double f=w[5]*T_a;
            f_temp[5][i][0]=f;
            f_temp[3][i][n-1]=f;
            if(i>0){
                f=w[2]*T_a;
                f_temp[2][i-1][0]=f;
                f_temp[0][i-1][n-1]=f;
            }
            if(i<nrows-1){
                f=w[8]*T_a;
                f_temp[8][i+1][0]=f;
                f_temp[6][i+1][n-1]=f;
            }
        }

        // Update values
        #pragma omp parallel for num_threads(thread_count) default(none) private(i,j,k) shared(n,nrows,T,T_i,T_l,T_b,T_a,w,myid,size,f,feq,f_temp,q,delx,lambda,alpha) reduction(+: local_error)
        for(i=0; i<nrows; i++){
            for(j=0; j<n; j++){
                T[i][j]=0;
                for(k=0; k<9; k++){
                    if((i==0 && myid==0) || (i==nrows && myid==size-1)){
                        break;
                    }
                    local_error+=pow(f[k][i][j]-f_temp[k][i][j],2); // Calculate error between successive iterations
                    f[k][i][j]=f_temp[k][i][j];
                    T[i][j]+=f[k][i][j]; // Calculate new temperature value
                }

                // Boundary conditions
                if(i==0 && myid==0){
                    T[i][j]=T_i; // Dirichlet
                    // T[i][j]=T_b; // Neumann and Robin
                }
                if(i==nrows-1 && myid==size-1){
                    T[i][j]=T_l; //Dirichlet 
                    // T[i][j]=T[i-1][j]-((q*delx)/lambda); //Neumann
                    // T[i][j]=T[i-1][j]*(lambda/(lambda+alpha*delx))+T_a*((alpha*delx)/(lambda+alpha*delx)); //Robin
                }

                // Calculate new equilibrium values
                for(int k=0; k<9; k++){
                    feq[k][i][j]=w[k]*T[i][j];
                }
            }
        }

        // store values to be sent to other processors in a buffer matrix
        double send_f_down[3][n], send_f_up[3][n], send_feq_down[3][n], send_feq_up[3][n];
        #pragma omp parallel for num_threads(thread_count) default(none) private(j,k) shared(n,nrows,send_f_up,send_feq_up,send_f_down,send_feq_down,f,feq)
        for(j=0; j<n; j++){
            for(k=6; k<9; k++){
                send_f_up[k-6][j]=f[k][nrows-1][j];
                send_feq_up[k-6][j]=feq[k][nrows-1][j];
            }
            for(k=0; k<3; k++){
                send_f_down[k][j]=f[k][0][j];
                send_feq_down[k][j]=feq[k][0][j];
            }
        }

        // Send boundary point values and recieve ghost point values only on the master thread
        #pragma omp master
        {
            if(myid==0){
                MPI_Send(send_f_up,3*n,MPI_DOUBLE,1,tag,MPI_COMM_WORLD);
                MPI_Recv(ghost_up,3*n,MPI_DOUBLE,1,tag,MPI_COMM_WORLD,&status);
                MPI_Send(send_feq_up,3*n,MPI_DOUBLE,1,tag,MPI_COMM_WORLD);
                MPI_Recv(ghost_up_eq,3*n,MPI_DOUBLE,1,tag,MPI_COMM_WORLD,&status);
            }
            else if(myid==size-1){
                MPI_Recv(ghost_down,3*n,MPI_DOUBLE,myid-1,tag,MPI_COMM_WORLD,&status);
                MPI_Send(send_f_down,3*n,MPI_DOUBLE,myid-1,tag,MPI_COMM_WORLD);
                MPI_Recv(ghost_down_eq,3*n,MPI_DOUBLE,myid-1,tag,MPI_COMM_WORLD,&status);
                MPI_Send(send_feq_down,3*n,MPI_DOUBLE,myid-1,tag,MPI_COMM_WORLD);
            }
            else{
                MPI_Recv(ghost_down,3*n,MPI_DOUBLE,myid-1,tag,MPI_COMM_WORLD,&status);
                MPI_Send(send_f_down,3*n,MPI_DOUBLE,myid-1,tag,MPI_COMM_WORLD);
                MPI_Recv(ghost_down_eq,3*n,MPI_DOUBLE,myid-1,tag,MPI_COMM_WORLD,&status);
                MPI_Send(send_feq_down,3*n,MPI_DOUBLE,myid-1,tag,MPI_COMM_WORLD);
                MPI_Send(send_f_up,3*n,MPI_DOUBLE,myid+1,tag,MPI_COMM_WORLD);
                MPI_Recv(ghost_up,3*n,MPI_DOUBLE,myid+1,tag,MPI_COMM_WORLD,&status);
                MPI_Send(send_feq_up,3*n,MPI_DOUBLE,myid+1,tag,MPI_COMM_WORLD);
                MPI_Recv(ghost_up_eq,3*n,MPI_DOUBLE,myid+1,tag,MPI_COMM_WORLD,&status);
            }
        }

        // Update global error values of all processors
        // Set an error threshold for breaking and print error values
        MPI_Allreduce(&local_error,&global_error,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
        global_error=sqrt(global_error);
        if(myid==0){
            printf("%d %.10f\n", it, global_error);
        }
        if(global_error<0.000001 && it!=1){
            break;
        }
    }

    // Print midplane temperature values
    if(myid==0){
        for(int j=0; j<nrows; j++){
            printf("%f ",T[j][n/2]);
        }
        for(int i=1; i<size; i++){
            if(i==size-1){
                nrows+=(n-nrows*size);
            }
            double temp[nrows];
            MPI_Recv(temp,nrows,MPI_DOUBLE,i,tag,MPI_COMM_WORLD,&status);
            for(int j=0; j<nrows; j++){
                printf("%f ",temp[j]);
            }
        }
    }
    else{
        double temp[nrows];
        for(int j=0; j<nrows; j++){
            temp[j]=T[j][n/2];
        }
        MPI_Send(temp,nrows,MPI_DOUBLE,0,tag,MPI_COMM_WORLD);
    }

    // Print temperature values at all points
    if(myid==0){
        for(int i=0; i<nrows; i++){
            for(int j=0; j<n; j++){
                printf("%f ",T[i][j]);
            }printf(";");
        }
        for(int i=1; i<size; i++){
            if(i==size-1){
                nrows+=(n-nrows*size);
            }
            double temp[nrows][n];
            MPI_Recv(temp,nrows*n,MPI_DOUBLE,i,tag,MPI_COMM_WORLD,&status);
            for(int i=0; i<nrows; i++){
                for(int j=0; j<n; j++){
                    printf("%f ",temp[i][j]);
                }printf(";");
            }
        }
    }
    else{
        MPI_Send(T,nrows*n,MPI_DOUBLE,0,tag,MPI_COMM_WORLD);
    }

    MPI_Finalize();
    // Print execution time
    double end=omp_get_wtime();
    printf("\nTime taken: %f\n",(end-start));
}