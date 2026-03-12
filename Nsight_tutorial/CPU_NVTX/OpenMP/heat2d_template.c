#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "nvToolsExt.h"

#define IDX(i,j) ((i)*ny + (j))

int main() {

    // --- SAME PARAMETERS AS ORIGINAL CODE ---
    const int nx = NX_PLACEHOLDER;
    const int ny = NY_PLACEHOLDER;
    const int nt = NT_PLACEHOLDER;
    const double Lx = 1.0;
    const double Ly = 1.0;
    const double alpha = 0.01;

    const double dx = Lx / (nx - 1);
    const double dy = Ly / (ny - 1);
    const double dt = 0.25 * fmin(dx*dx, dy*dy) / alpha;

    // ---- Memory allocation ----
    nvtxRangePushA("Memory Allocation");
    double * restrict u     = (double*) calloc(nx * ny, sizeof(double));
    double * restrict u_new = (double*) calloc(nx * ny, sizeof(double));
    nvtxRangePop();

    // ---- Initialization ----
    nvtxRangePushA("Initialization");

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            double x = i * dx;
            double y = j * dy;
            u[IDX(i,j)] = exp(-50.0 * ((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5)));
        }
    }

    nvtxRangePop();

    // ---- Time stepping ----
    nvtxRangePushA("Time Stepping");
    double t_start = omp_get_wtime();

    #pragma omp parallel
    for (int n = 0; n < nt; n++) {

        #pragma omp for
        for (int i = 1; i < nx-1; i++) {
	    #pragma ivdep
            for (int j = 1; j < ny-1; j++) {

                u_new[IDX(i,j)] = u[IDX(i,j)] +
                    alpha * dt * (
                        (u[IDX(i+1,j)] - 2*u[IDX(i,j)] + u[IDX(i-1,j)])/(dx*dx)
                      + (u[IDX(i,j+1)] - 2*u[IDX(i,j)] + u[IDX(i,j-1)])/(dy*dy)
                    );
            }
        }

        // swap arrays
	#pragma omp single
	{
        double *tmp = u;
        u = u_new;
        u_new = tmp;
	}
    }
    double t_end = omp_get_wtime();

    nvtxRangePop();

    printf("Center temperature: %f\n", u[IDX(nx/2, ny/2)]);

    free(u);
    free(u_new);
    double elapsed = t_end - t_start;
    double points = (double)nx * ny * nt;
    printf("Elapsed time (s): %f\n", elapsed);
    printf("Grid: %d x %d, Steps: %d\n", nx, ny, nt);
    printf("Points updated: %.3e\n", points);
    printf("Throughput (updates/sec): %.3e\n", points / elapsed);
    printf("OpenMP max threads: %d\n", omp_get_max_threads());
    return 0;
}
