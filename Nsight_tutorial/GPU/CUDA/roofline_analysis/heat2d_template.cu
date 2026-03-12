#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define IDX(i,j) ((i)*ny + (j))

// ================= CUDA KERNEL =================
__global__
void heat_step(double *u, double *u_new,
               int nx, int ny,
               double dx, double dy,
               double alpha, double dt)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > 0 && i < nx-1 && j > 0 && j < ny-1) {

        int idx = i * ny + j;

        u_new[idx] = u[idx] +
            alpha * dt * (
                (u[(i+1)*ny + j] - 2.0*u[idx] + u[(i-1)*ny + j])/(dx*dx)
              + (u[i*ny + (j+1)] - 2.0*u[idx] + u[i*ny + (j-1)])/(dy*dy)
            );
    }
}

// ================= MAIN =================
int main() {

    const int nx = NX_PLACEHOLDER;
    const int ny = NY_PLACEHOLDER;
    const int nt = NT_PLACEHOLDER;
    const int nb = NB_PLACEHOLDER;

    const double Lx = 1.0;
    const double Ly = 1.0;
    const double alpha = 0.01;

    const double dx = Lx / (nx - 1);
    const double dy = Ly / (ny - 1);
    const double dt = 0.25 * fmin(dx*dx, dy*dy) / alpha;
    size_t npoints = nx * ny;
    size_t size = npoints * sizeof(double);

    // -------- Host memory --------
    double *h_u = (double*) calloc(nx * ny, sizeof(double));

    // -------- Device memory --------
    double *d_u, *d_u_new;
    cudaMalloc(&d_u, size);
    cudaMalloc(&d_u_new, size);

    // -------- Initialization (host) --------

    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            double x = i * dx;
            double y = j * dy;
            h_u[IDX(i,j)] =
                exp(-50.0 * ((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5)));
        }
    }

    cudaMemcpy(d_u, h_u, size, cudaMemcpyHostToDevice);

    // -------- Kernel configuration --------
    //dim3 block(16,16);
    dim3 block(nb,nb);
    dim3 grid((ny + block.x - 1)/block.x,
              (nx + block.y - 1)/block.y);

    // -------- Time stepping --------

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int n = 0; n < nt; n++) {

        heat_step<<<grid, block>>>(d_u, d_u_new,
                                   nx, ny,
                                   dx, dy,
                                   alpha, dt);

        // swap device pointers
        double *tmp = d_u;
        d_u = d_u_new;
        d_u_new = tmp;
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);


    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double elapsed = milliseconds / 1000.0;

    // -------- Copy result back --------
    cudaMemcpy(h_u, d_u, size, cudaMemcpyDeviceToHost);

    printf("Center temperature: %f\n", h_u[IDX(nx/2, ny/2)]);
    printf("Elapsed time (s): %f\n", elapsed);
    printf("Grid: %d x %d, Steps: %d, block: %d\n", nx, ny, nt, nb);
    double points = (double)nx * ny * nt;
    printf("Points updated: %.3e\n", points);
    printf("Throughput (updates/sec): %.3e\n", points / elapsed);

    // -------- Cleanup --------
    free(h_u);
    cudaFree(d_u);
    cudaFree(d_u_new);

    return 0;
}
