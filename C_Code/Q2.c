#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <lapacke.h>
#include <cblas.h>
#include <math.h>
#include <string.h>

#define IDX(i, j, n) ((i) * (n) + (j))  // Convert 2D indices to 1D

// TSQR computation
void TSQR(double* A, int m, int n, double* Q, double* R, MPI_Comm comm) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    int block_size = m / size;  // Number of rows per process
    double* local_A = (double*)malloc(block_size * n * sizeof(double));
    double* local_Q = (double*)malloc(block_size * n * sizeof(double));
    double* local_R = (double*)malloc(n * n * sizeof(double));

    // Step 1: Scatter A among processes
    MPI_Scatter(A, block_size * n, MPI_DOUBLE, local_A, block_size * n, MPI_DOUBLE, 0, comm);

    // Step 2: Compute local QR decomposition
    double tau[n];
    LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, block_size, n, local_A, n, tau);

    // Extract upper triangular R matrix
    memset(local_R, 0, n * n * sizeof(double));
    for (int i = 0; i < n; i++)
        for (int j = 0; j <= i; j++)
            local_R[IDX(j, i, n)] = local_A[IDX(j, i, n)];

    // Compute Q from QR decomposition
    LAPACKE_dorgqr(LAPACK_ROW_MAJOR, block_size, n, n, local_A, n, tau);
    memcpy(local_Q, local_A, block_size * n * sizeof(double));

    // Step 3: Gather all R blocks to root process
    double* R_blocks = NULL;
    if (rank == 0) R_blocks = (double*)malloc(size * n * n * sizeof(double));
    MPI_Gather(local_R, n * n, MPI_DOUBLE, R_blocks, n * n, MPI_DOUBLE, 0, comm);

    double* Q2 = NULL;
    if (rank == 0) {
        double tau_global[n];
        Q2 = (double*)malloc(size * n * n * sizeof(double));

        // Global QR decomposition on stacked R blocks
        LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, size * n, n, R_blocks, n, tau_global);
        memcpy(R, R_blocks, n * n * sizeof(double));
        LAPACKE_dorgqr(LAPACK_ROW_MAJOR, size * n, n, n, R_blocks, n, tau_global);
        memcpy(Q2, R_blocks, size * n * n * sizeof(double));
    }

    // Broadcast final R matrix to all processes
    MPI_Bcast(R, n * n, MPI_DOUBLE, 0, comm);

    // Step 4: Scatter Q2 to processes
    double* Q2_sub = (double*)malloc(n * n * sizeof(double));
    MPI_Scatter(Q2, n * n, MPI_DOUBLE, Q2_sub, n * n, MPI_DOUBLE, 0, comm);

    // Step 5: Compute final local Q
    double* local_Q_updated = (double*)malloc(block_size * n * sizeof(double));
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, block_size, n, n,
                1.0, local_Q, n, Q2_sub, n, 0.0, local_Q_updated, n);

    // Step 6: Gather final Q matrix from all processes
    MPI_Allgather(local_Q_updated, block_size * n, MPI_DOUBLE, Q, block_size * n, MPI_DOUBLE, comm);

    free(local_A);
    free(local_Q);
    free(local_R);
    free(local_Q_updated);
    free(Q2_sub);
    if (rank == 0) free(R_blocks);
}

// Compute orthogonality error ||QᵀQ - I||
double check_orthogonality(double* Q, int m, int n) {
    double* QtQ = (double*)malloc(n * n * sizeof(double));
    memset(QtQ, 0, n * n * sizeof(double));

    cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, n, m,
                1.0, Q, n, Q, n, 0.0, QtQ, n);

    double error = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            error += (QtQ[IDX(i, j, n)] - (i == j ? 1.0 : 0.0)) * (QtQ[IDX(i, j, n)] - (i == j ? 1.0 : 0.0));

    free(QtQ);
    return sqrt(error);
}

// Compute QR reconstruction error ||QR - A||
double check_qr_reconstruction(double* Q, double* R, double* A, int m, int n) {
    double* QR = (double*)malloc(m * n * sizeof(double));
    memset(QR, 0, m * n * sizeof(double));

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, n,
                1.0, Q, n, R, n, 0.0, QR, n);

    double error = 0.0;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            error += (QR[IDX(i, j, n)] - A[IDX(i, j, n)]) * (QR[IDX(i, j, n)] - A[IDX(i, j, n)]);

    free(QR);
    return sqrt(error);
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int m = 200, n = 10;
    double A[m * n], Q[m * n], R[n * n];

    // Initialize matrix A with random values
    if (rank == 0) {
        srand(42);
        for (int i = 0; i < m * n; i++)
            A[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }

    TSQR(A, m, n, Q, R, MPI_COMM_WORLD);

    // Compute and print errors on root process
    if (rank == 0) {
        double ortho_error = check_orthogonality(Q, m, n);
        double qr_error = check_qr_reconstruction(Q, R, A, m, n);

        printf("Orthogonality error: %.16e\n", ortho_error);
        printf("QR correctness error: %.16e\n", qr_error);
    }

    MPI_Finalize();
    return 0;
}
