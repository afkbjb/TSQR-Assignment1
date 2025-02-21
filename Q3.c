#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <lapacke.h>
#include <cblas.h>
#include <math.h>
#include <string.h>

#define IDX(i, j, n) ((i) * (n) + (j))

void TSQR(double* A, int m, int n, double* Q, double* R, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int block_size = (m + size - 1) / size;

    // Print warning only on the root process
    if (rank == 0 && block_size < n) {
        printf("Warning: Skipping m=%d, n=%d, size=%d -> block_size=%d is too small!\n",
               m, n, size, block_size);
    }

    // Broadcast the "skip" status
    int skip = (block_size < n) ? 1 : 0;
    MPI_Bcast(&skip, 1, MPI_INT, 0, comm);
    if (skip) return; // Skip computation without terminating MPI processes

    double* local_A = (double*)malloc(block_size * n * sizeof(double));
    double* local_Q = (double*)malloc(block_size * n * sizeof(double));
    double* local_R = (double*)malloc(n * n * sizeof(double));

    MPI_Scatter(A, block_size * n, MPI_DOUBLE, local_A, block_size * n, MPI_DOUBLE, 0, comm);
    MPI_Barrier(comm);

    double tau[n];
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, block_size, n, local_A, n, tau);

    memset(local_R, 0, n * n * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++)
            local_R[IDX(j, i, n)] = local_A[IDX(j, i, n)];

    LAPACKE_dorgqr(LAPACK_ROW_MAJOR, block_size, n, n, local_A, n, tau);
    memcpy(local_Q, local_A, block_size * n * sizeof(double));

    double* R_blocks = NULL;
    if (rank == 0) R_blocks = (double*)malloc(size * n * n * sizeof(double));
    MPI_Gather(local_R, n * n, MPI_DOUBLE, R_blocks, n * n, MPI_DOUBLE, 0, comm);

    double* Q2 = NULL;
    if (rank == 0) {
        double tau_global[n];
        Q2 = (double*)malloc(size * n * n * sizeof(double));
        LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, size * n, n, R_blocks, n, tau_global);
        memcpy(R, R_blocks, n * n * sizeof(double));
        LAPACKE_dorgqr(LAPACK_ROW_MAJOR, size * n, n, n, R_blocks, n, tau_global);
        memcpy(Q2, R_blocks, size * n * n * sizeof(double));
    }

    MPI_Bcast(R, n * n, MPI_DOUBLE, 0, comm);

    double* Q2_sub = (double*)malloc(n * n * sizeof(double));
    MPI_Scatter(Q2, n * n, MPI_DOUBLE, Q2_sub, n * n, MPI_DOUBLE, 0, comm);

    double* local_Q_updated = (double*)malloc(block_size * n * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, block_size, n, n,
                1.0, local_Q, n, Q2_sub, n, 0.0, local_Q_updated, n);

    MPI_Allgather(local_Q_updated, block_size * n, MPI_DOUBLE, Q, block_size * n, MPI_DOUBLE, comm);

    free(local_A);
    free(local_Q);
    free(local_R);
    free(local_Q_updated);
    free(Q2_sub);
    if (rank == 0) free(R_blocks);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int m_values[] = {100, 500, 1000, 5000, 10000};  // Test different m values
    int n_values[] = {5, 10, 20, 50};  // Test different n values

    FILE* fp = NULL;
    if (rank == 0) fp = fopen("scaling_results.txt", "w");

    for (int i = 0; i < sizeof(m_values) / sizeof(m_values[0]); i++) {
        for (int j = 0; j < sizeof(n_values) / sizeof(n_values[0]); j++) {
            int m = m_values[i];
            int n = n_values[j];

            if (m < n) continue; // Skip cases where m < n

            double* A = NULL;
            double* Q = (double*)malloc(m * n * sizeof(double));
            double* R = (double*)malloc(n * n * sizeof(double));

            if (rank == 0) {
                A = (double*)malloc(m * n * sizeof(double));
                srand(42);
                for (int k = 0; k < m * n; k++)
                    A[k] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
            }

            MPI_Barrier(MPI_COMM_WORLD);
            double start_time = MPI_Wtime();
            TSQR(A, m, n, Q, R, MPI_COMM_WORLD);
            double end_time = MPI_Wtime();

            if (rank == 0) {
                double time_taken = end_time - start_time;
                
                // Print progress information to stderr to avoid writing to scaling_results.txt
                fprintf(stderr, "Processing: m=%d, n=%d, Time: %.6f sec\n", m, n, time_taken);
            
                // Write only m, n, and time results to the file
                fprintf(fp, "%d %d %.6f\n", m, n, time_taken);
            
                free(A);
            }                     

            free(Q);
            free(R);
        }
    }

    if (rank == 0) fclose(fp);
    MPI_Finalize();
    return 0;
}
