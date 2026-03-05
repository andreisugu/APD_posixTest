#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int numprocs, procid;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);

    int m = argc > 1 ? atoi(argv[1]) : 100;
    if (m < 100) m = 100;

    double start = MPI_Wtime();
    srand(time(NULL) + procid * 1000);

    long long sum = 0;
    printf("Process %d numbers: ", procid);
    for (int i = 0; i < m; i++) {
        int val = rand() % 1001;
        printf("%d ", val);
        sum += val;
    }
    printf("\nProcess %d: sum = %lld\n", procid, sum);

    double elapsed = MPI_Wtime() - start;
    printf("Process %d: time = %f s\n", procid, elapsed);

    if (procid != 0) {
        MPI_Send(&elapsed, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    } else {
        printf("\n=== Timing Summary (m=%d, %d procs) ===\n", m, numprocs);
        printf("Process 0: %f s\n", elapsed);
        for (int i = 1; i < numprocs; i++) {
            double t;
            MPI_Recv(&t, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
            printf("Process %d: %f s\n", i, t);
        }
    }

    MPI_Finalize();
    return 0;
}
