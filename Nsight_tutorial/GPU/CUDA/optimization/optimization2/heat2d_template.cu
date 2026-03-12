#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define BLOCK 32      // 16x16 threads per block
#define HALO 1        // halo for 5-point stencil
#define T_BLOCK TBLOCK     // number of timesteps per tile

#define IDX(i,j,ny) ((i)*(ny) + (j))

// ================= CUDA KERNEL (Shared Memory + Temporal Blocking) =================
__global__
void heat_step_temporal(double *u, double *u_new,
                        int nx, int ny,
                        double dx, double dy,
                        double alpha, double dt)
{
    __shared__ double tile[BLOCK + 2*HALO][BLOCK + 2*HALO];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i_global = blockIdx.y * BLOCK + ty;
    int j_global = blockIdx.x * BLOCK + tx;

    // Only process valid interior points
    if(i_global >= nx || j_global >= ny)
        return;

    // Load tile + halo into shared memory
    // Central points
    tile[ty+HALO][tx+HALO] = u[IDX(i_global,j_global,ny)];

    // Halos
    if(ty < HALO) {
        if(i_global >= HALO)
            tile[ty][tx+HALO] = u[IDX(i_global-HALO,j_global,ny)];       // top
        if(i_global + BLOCK < nx)
            tile[ty+BLOCK+HALO][tx+HALO] = u[IDX(i_global+BLOCK,j_global,ny)]; // bottom
    }
    if(tx < HALO) {
        if(j_global >= HALO)
            tile[ty+HALO][tx] = u[IDX(i_global,j_global-HALO,ny)];       // left
        if(j_global + BLOCK < ny)
            tile[ty+HALO][tx+BLOCK+HALO] = u[IDX(i_global,j_global+BLOCK,ny)]; // right
    }

    __syncthreads();

    // Local buffers for temporal blocking
    double center, up, down, left, right;
    double tmp[2];   // double buffer for alternating timesteps
    tmp[0] = tile[ty+HALO][tx+HALO]; // initial value
    int curr = 0, next = 1;

    // Compute T_BLOCK timesteps in shared memory
    for(int t=0; t<T_BLOCK; t++) {

        // Load neighbors from shared memory (simple stencil)
        up    = tile[ty+HALO-1][tx+HALO];
        down  = tile[ty+HALO+1][tx+HALO];
        left  = tile[ty+HALO][tx+HALO-1];
        right = tile[ty+HALO][tx+HALO+1];

        tmp[next] = tmp[curr] + alpha*dt * ((up - 2.0*tmp[curr] + down)/(dx*dx)
                                          + (left - 2.0*tmp[curr] + right)/(dy*dy));

        // Swap buffers
        int tmp_idx = curr; curr = next; next = tmp_idx;

        __syncthreads();
    }

    // Write final value back to global memory
    if(i_global>0 && i_global<nx-1 && j_global>0 && j_global<ny-1)
        u_new[IDX(i_global,j_global,ny)] = tmp[curr];
}

// ================= MAIN =================
int main() {

    const int nx = NX_PLACEHOLDER;
    const int ny = NY_PLACEHOLDER;
    const int nt = NT_PLACEHOLDER;

    const double Lx = 1.0;
    const double Ly = 1.0;
    const double alpha = 0.01;

    const double dx = Lx / (nx-1);
    const double dy = Ly / (ny-1);
    const double dt = 0.25 * fmin(dx*dx, dy*dy) / alpha;

    size_t npoints = (size_t)nx*ny;
    size_t size = npoints*sizeof(double);

    // Host memory
    double *h_u = (double*) calloc(npoints, sizeof(double));

    // Device memory
    double *d_u, *d_u_new;
    cudaMalloc(&d_u, size);
    cudaMalloc(&d_u_new, size);

    // Initialization
    for(int i=0; i<nx; i++)
        for(int j=0; j<ny; j++){
            double x = i*dx;
            double y = j*dy;
            h_u[IDX(i,j,ny)] = exp(-50.0*((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5)));
        }

    cudaMemcpy(d_u, h_u, size, cudaMemcpyHostToDevice);

    // Kernel configuration
    dim3 block(BLOCK, BLOCK);
    dim3 grid((ny + BLOCK - 1)/BLOCK, (nx + BLOCK - 1)/BLOCK);

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int steps_remaining = nt;
    while(steps_remaining > 0){
        int t_steps = (steps_remaining >= T_BLOCK) ? T_BLOCK : steps_remaining;

        heat_step_temporal<<<grid, block>>>(d_u, d_u_new, nx, ny,
                                            dx, dy, alpha, dt);

        // swap pointers
        double *tmp = d_u;
        d_u = d_u_new;
        d_u_new = tmp;

        steps_remaining -= t_steps;
    }

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double elapsed = milliseconds/1000.0;

    // Copy back
    cudaMemcpy(h_u, d_u, size, cudaMemcpyDeviceToHost);

    printf("Center temperature: %f\n", h_u[IDX(nx/2, ny/2,ny)]);
    printf("Elapsed time (s): %f\n", elapsed);
    printf("Grid: %d x %d, Steps: %d, Block: %d\n", nx, ny, nt, BLOCK);
    double points = (double)nx*ny*nt;
    printf("Points updated: %.3e\n", points);
    printf("Throughput (updates/sec): %.3e\n", points/elapsed);

    // Cleanup
    free(h_u);
    cudaFree(d_u);
    cudaFree(d_u_new);

    return 0;
}
