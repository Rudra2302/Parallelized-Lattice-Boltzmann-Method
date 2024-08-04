#include <stdio.h> 
#include <string.h> 
#include <stddef.h> 
#include <stdlib.h>
#include <math.h>
#include <time.h>
#define pi 3.14159

// Serial code for LBM

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

int main(){

    // Declare variables
    clock_t start=clock();
    double delx=0.0001, dely=0.0001, delt=0.1;
    double T_i=0, T_l=100, T_a=25, T_b=150, alpha=10, tau=0.1, q=1000, lambda=35;
    int n=10; // grid size
    double f[9][n+1][n+1], feq[9][n+1][n+1], T[n+1][n+1]; // f for function value, feq for equilibrium value, T for temperature
    double w[9], c[2][9]; // c for direction, w for directional weight

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
    w[4]=0;

    // Initialize directions
    for(int i=0; i<=2; i++){
        for(int j=0; j<=2; j++){
            int k=i*3+j;
            c[0][k]=i-1; c[1][k]=j-1;
        }
    }

    // Initialize temperature values
    for(int i=0; i<n+1; i++){
        for(int j=0; j<n+1; j++){

            T[i][j]=T_i; //Dirichlet

            //Neumann and Robin
            // if(i==0){ 
            //     T[i][j]=T_b;
            // }
            // else{
            //     T[i][j]=0;
            // }

            // Initialize f and feq values 
            for(int k=0; k<9; k++){
                feq[k][i][j]=w[k]*T[i][j];
                f[k][i][j]=feq[k][i][j];
            }
        }
    }

    // Run for the time required
    for(int it=1; it<=50000; it++){
        // Declare a temperary function matrix
        double f_temp[9][n+1][n+1], error=0;

        // calculate new f values
        for(int i=0; i<n+1; i++){
            for(int j=0; j<n+1; j++){
                // Run loop for the 9 directions
                for(int k=0; k<9; k++){
                    int ii=i+c[0][k], jj=j+c[1][k];
                    if(ii<0 || ii>n || jj<0 || jj>n){
                        continue;
                    }
                    f_temp[k][ii][jj]=f[k][i][j]-(delt/tau)*(f[k][i][j]-feq[k][i][j]); // Calculate function value
                }
            }
        }

        // Impact on points just outside boundary on the boundary points of j=0 and j=n-1
        for(int i=0; i<n+1; i++){
            double f=w[5]*T_a;
            f_temp[5][i][0]=f;
            f_temp[3][i][n]=f;
            if(i>0){
                f=w[2]*T_a;
                f_temp[2][i-1][0]=f;
                f_temp[0][i-1][n]=f;
            }
            if(i<n){
                f=w[8]*T_a;
                f_temp[8][i+1][0]=f;
                f_temp[6][i+1][n]=f;
            }
        }

        // Update values
        for(int i=0; i<n+1; i++){
            for(int j=0; j<n+1; j++){
                T[i][j]=0;
                for(int k=0; k<9; k++){
                    if(i==0 || i==n){
                        break;
                    }
                    error+=pow(f[k][i][j]-f_temp[k][i][j],2); // Calculate error between successive iterations
                    f[k][i][j]=f_temp[k][i][j];
                    T[i][j]+=f[k][i][j]; // Calculate new temperature value
                }

                // Boundary conditions
                if(i==0){
                    T[i][j]=T_i; // Dirichlet
                    // T[i][j]=T_b; // Neumann and Robin
                }
                if(i==n){
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

        // Set an error threshold for breaking and print error values
        error=sqrt(error);
        printf("%d %.10f\n",it, error);
        if(error<0.000001 && it!=1){
            break;
        }
    }

    // Print temperature values of whole grid
    for(int i=0; i<n+1; i++){
        for(int j=0; j<n+1; j++){
            printf("%f ",T[i][j]);
        }printf(";");
    }
    // Temperature values on the midplane
    for(int i=0; i<n+1; i++){
        printf("%f ",T[i][n/2]);
    }
    // Print time of execution
    clock_t end=clock();
    printf("\nTime taken: %ld\n",(end-start));
}