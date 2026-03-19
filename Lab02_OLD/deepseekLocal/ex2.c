#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Read matrix dimensions from root process
    int m, n;
    double *matrixA, *matrixB, *result;

    if (rank == 0) {
        printf("Enter matrix dimensions (m x n): ");
        scanf("%d %d", &m, &n);
        printf("Enter matrix A:\n");
        matrixA = (double *)malloc(m * n * sizeof(double));
        for (int i = 0; i < m * n; i++) {
            scanf("%lf", &matrixA[i]);
        }
        printf("Enter matrix B:\n");
        matrixB = (double *)malloc(m * n * sizeof(double));
        for (int i = 0; i < m * n; i++) {
            scanf("%lf", &matrixB[i]);
        }
    }

    // Broadcast matrix dimensions to all processes
    MPI_Bcast(&m, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calculate the number of elements each process will handle
    int num_elements = (m / size) * n;
    int remainder = m % size;

    // Allocate memory for local matrices
    double *localA = (double *)malloc(num_elements * sizeof(double));
    double *localB = (double *)malloc(num_elements * sizeof(double));
    double *localResult = (double *)malloc(num_elements * sizeof(double));

    // Scatter the matrices to all processes
    MPI_Scatter(matrixA, num_elements, MPI_DOUBLE, localA, num_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(matrixB, num_elements, MPI_DOUBLE, localB, num_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Perform element-wise addition on the local matrices
    for (int i = 0; i < num_elements; i++) {
        localResult[i] = localA[i] + localB[i];
    }

    // Gather the results back to the root process
    MPI_Gather(localResult, num_elements, MPI_DOUBLE, result, num_elements, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Free memory and finalize
    free(localA);
    free(localB);
    free(localResult);

    if (rank == 0) {
        printf("Result matrix:\n");
        for (int i = 0; i < m * n; i++) {
            printf("%lf ", result[i]);
            if ((i + 1) % n == 0) {
                printf("\n");
            }
        }
    }

    MPI_Finalize();
    return 0;
}