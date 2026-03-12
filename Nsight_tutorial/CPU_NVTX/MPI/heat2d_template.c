#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#define IDX(i,j) ((i)*ny + (j))

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ----- Global problem size -----
    const int nx = NX_PLACEHOLDER;
    const int ny = NY_PLACEHOLDER;
    const int nt = NT_PLACEHOLDER;
    const double Lx = 1.0;
    const double Ly = 1.0;
    const double alpha = 0.01;

    const double dx = Lx / (nx - 1);
    const double dy = Ly / (ny - 1);
    const double dt = 0.25 * fmin(dx*dx, dy*dy) / alpha;

    // ----- 1D row decomposition -----
    int local_nx = nx / size;
    int remainder = nx % size;

    if (rank < remainder)
        local_nx++;

    // determine starting global row
    int start = rank * (nx / size) + (rank < remainder ? rank : remainder);

    // allocate with 2 ghost rows
    int local_size = (local_nx + 2) * ny;

    double *u     = calloc(local_size, sizeof(double));
    double *u_new = calloc(local_size, sizeof(double));

    // ----- Initialization -----
    for (int i = 1; i <= local_nx; i++) {
        int global_i = start + i - 1;
        for (int j = 0; j < ny; j++) {
            double x = global_i * dx;
            double y = j * dy;
            u[IDX(i,j)] =
                exp(-50.0 * ((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5)));
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    // ----- Time stepping -----
    for (int n = 0; n < nt; n++) {

        // ---- Halo exchange ----
        if (rank > 0)
            MPI_Sendrecv(&u[IDX(1,0)], ny, MPI_DOUBLE, rank-1, 0,
                         &u[IDX(0,0)], ny, MPI_DOUBLE, rank-1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (rank < size-1)
            MPI_Sendrecv(&u[IDX(local_nx,0)], ny, MPI_DOUBLE, rank+1, 0,
                         &u[IDX(local_nx+1,0)], ny, MPI_DOUBLE, rank+1, 0,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        // ---- Update interior ----
        for (int i = 1; i <= local_nx; i++) {
            for (int j = 1; j < ny-1; j++) {

                u_new[IDX(i,j)] = u[IDX(i,j)] +
                    alpha * dt * (
                        (u[IDX(i+1,j)] - 2*u[IDX(i,j)] + u[IDX(i-1,j)])/(dx*dx)
                      + (u[IDX(i,j+1)] - 2*u[IDX(i,j)] + u[IDX(i,j-1)])/(dy*dy)
                    );
            }
        }

        // swap arrays
        double *tmp = u;
        u = u_new;
        u_new = tmp;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    // ----- Print center value (rank that owns it) -----
    int center_i = nx / 2;

    if (center_i >= start && center_i < start + local_nx) {
        int local_i = center_i - start + 1;
        printf("Center temperature: %f\n",
               u[IDX(local_i, ny/2)]);
    }

    if (rank == 0) {
        double elapsed = t_end - t_start;
        double points = (double)nx * ny * nt;
	printf("Elapsed time (s): %f\n", elapsed);
        printf("Grid: %d x %d, Steps: %d\n", nx, ny, nt);
        printf("Points updated: %.3e\n", points);
        printf("Throughput (updates/sec): %.3e\n", points / elapsed);
        printf("MPI tasks: %d\n", size);
    }

    free(u);
    free(u_new);

    MPI_Finalize();
    return 0;
}
