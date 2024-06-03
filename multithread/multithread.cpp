#include <iostream>
#include <boost/program_options.hpp>
#include <stdio.h>
#include <chrono>
#include <math.h>
#include <cstring>
#include <omp.h>

#define OFFSET(x, y, m) (((x)*(m)) + (y))

using namespace std;
namespace opt = boost::program_options;

void initialize(double *A, double *Anew, int m, int n)
{
    memset(A, 0, n * m * sizeof(double));
    memset(Anew, 0, n * m * sizeof(double));

    for(int i = 0; i < m; i++){
        A[i] = 1.0;
        Anew[i] = 1.0;
    }
}

void deallocate(double *A, double *Anew)
{
    free(A);
    free(Anew);
}

int main(int argc, char *argv[])
{
    opt::options_description desc("All options");

    desc.add_options()
            ("precision", opt::value<double>())
            ("size", opt::value<int>())
            ("iter", opt::value<int>())
            ;

    opt::variables_map vm;
    opt::store(opt::parse_command_line(argc, argv, desc), vm);
    opt::notify(vm);
    const int n = vm["size"].as<int>();
    const int m = vm["size"].as<int>();
    const int iter_max = vm["iter"].as<int>();

    const double tol = vm["precision"].as<double>();

    double *A = new double[n * m];
    double *Anew = new double[n * m];

    initialize(A, Anew, m, n);

    cout << "Jacobi relaxation Calculation: " << n << " x " << m << " mesh" << endl;

    auto start = std::chrono::steady_clock::now();
    int iter = 0;

    double err = 1.0;

    {
        while (err > tol && iter < iter_max)
        {
            err = 0.0;
#pragma acc parallel loop reduction(max:err)
            for( int j = 1; j < n-1; j++)
            {
#pragma acc loop
                for( int i = 1; i < m-1; i++ )
                {
                    Anew[OFFSET(j, i, m)] = 0.25 * ( A[OFFSET(j, i+1, m)] + A[OFFSET(j, i-1, m)]
                                                     + A[OFFSET(j-1, i, m)] + A[OFFSET(j+1, i, m)]);
                    err = fmax( err, fabs(Anew[OFFSET(j, i, m)] - A[OFFSET(j, i , m)]));
                }
            }

#pragma acc parallel loop
            for( int j = 1; j < n-1; j++)
            {
#pragma acc loop
                for( int i = 1; i < m-1; i++ )
                {
                    A[OFFSET(j, i, m)] = Anew[OFFSET(j, i, m)];
                }
            }

            if (iter % 100 == 0)
                printf("%5d, %0.6f\n", iter, err);

            iter++;
        }
    }

    auto end = std::chrono::steady_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf(" total: %d mls\n", duration.count());

    deallocate(A, Anew);

    return 0;
}