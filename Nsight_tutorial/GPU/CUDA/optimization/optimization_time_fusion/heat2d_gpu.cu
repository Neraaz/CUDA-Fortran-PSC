#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define IDX(i,j,ny) ((i)*(ny) + (j))

// ================= CUDA KERNEL =================
// Compute 'steps_per_kernel' timesteps per launch
__global__
void heat_step_fused(double *u, double *u_new,
                     int nx, int ny,
                     double dx, double dy,
                     double alpha, double dt,
                     int steps_per_kernel)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i == 0 || i >= nx-1 || j == 0 || j >= ny-1)
        return; // skip boundaries

    int idx = IDX(i,j,ny);

    double center;
    double left, right, top, bottom;

    // Local registers
    center = u[idx];

    for (int t=0; t < steps_per_kernel; t++) {

        // Read neighbors from global memory for first step
        if (t == 0) {
            left   = u[IDX(i,j-1,ny)];
            right  = u[IDX(i,j+1,ny)];
            top    = u[IDX(i-1,j,ny)];
            bottom = u[IDX(i+1,j,ny)];
        }

        // Compute update
        double u_new_val = center +
            alpha * dt * (
                (bottom - 2.0*center + top)/(dx*dx) +
                (right  - 2.0*center + left)/(dy*dy)
            );

        // Shift registers for next timestep
        center = u_new_val;

        // Update neighbors for next iteration if needed
        if (t < steps_per_kernel-1) {
            left   = u[IDX(i,j-1,ny)];
            right  = u[IDX(i,j+1,ny)];
            top    = u[IDX(i-1,j,ny)];
            bottom = u[IDX(i+1,j,ny)];
        }
    }

    // Write final value to global memory
    u_new[idx] = center;
}

// ================= MAIN =================
int main() {

    const int nx = 30000;
    const int ny = 30000;
    const int nt = 5000;
    const int nb = 16;
    const int steps_per_kernel = 16; // fuse 4 timesteps per kernel

    const double Lx = 1.0;
    const double Ly = 1.0;
    const double alpha = 0.01;

    const double dx = Lx / (nx - 1);
    const double dy = Ly / (ny - 1);
    const double dt = 0.25 * fmin(dx*dx, dy*dy) / alpha;

    size_t npoints = nx * ny;
    size_t size = npoints * sizeof(double);

    // -------- Host memory --------
    double *h_u = (double*) calloc(npoints, sizeof(double));

    // -------- Device memory --------
    double *d_u, *d_u_new;
    cudaMalloc(&d_u, size);
    cudaMalloc(&d_u_new, size);

    // -------- Initialization --------
    for (int i = 0; i < nx; i++) {
        for (int j = 0; j < ny; j++) {
            double x = i * dx;
            double y = j * dy;
            h_u[IDX(i,j,ny)] = exp(-50.0 * ((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5)));
        }
    }

    cudaMemcpy(d_u, h_u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_u_new, d_u, size, cudaMemcpyDeviceToDevice);

    // -------- Kernel configuration --------
    dim3 block(nb, nb);
    dim3 grid((ny + block.x - 1)/block.x,
              (nx + block.y - 1)/block.y);

    // -------- Time stepping --------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int steps_done = 0;
    while (steps_done < nt) {
        int steps_now = steps_per_kernel;
        if (steps_done + steps_now > nt)
            steps_now = nt - steps_done;

        heat_step_fused<<<grid, block>>>(d_u, d_u_new,
                                         nx, ny,
                                         dx, dy,
                                         alpha, dt,
                                         steps_now);

        // Swap device pointers
        double *tmp = d_u;
        d_u = d_u_new;
        d_u_new = tmp;

        steps_done += steps_now;
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double elapsed = milliseconds / 1000.0;

    // -------- Copy result back --------
    cudaMemcpy(h_u, d_u, size, cudaMemcpyDeviceToHost);

    printf("Center temperature: %f\n", h_u[IDX(nx/2, ny/2, ny)]);
    printf("Elapsed time (s): %f\n", elapsed);
    printf("Grid: %d x %d, Steps: %d, block: %d, steps_per_kernel: %d\n",
           nx, ny, nt, nb, steps_per_kernel);
    double points = (double)nx * ny * nt;
    printf("Points updated: %.3e\n", points);
    printf("Throughput (updates/sec): %.3e\n", points / elapsed);

    // -------- Cleanup --------
    free(h_u);
    cudaFree(d_u);
    cudaFree(d_u_new);

    return 0;
}
