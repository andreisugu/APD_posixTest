#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int is_prime(int n) {
    if (n < 2) return 0;
    if (n == 2) return 1;
    if (n % 2 == 0) return 0;
    for (int i = 3; i * i <= n; i += 2)
        if (n % i == 0) return 0;
    return 1;
}

int main(int argc, char *argv[]) {
    int numprocs, procid;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);

    int N = argc > 1 ? atoi(argv[1]) : 100;

    /* Cyclic distribution: process i checks 2+i, 2+i+numprocs, ... */
    int local_count = 0;
    for (int num = 2 + procid; num < N; num += numprocs)
        if (is_prime(num))
            local_count++;

    if (procid != 0) {
        MPI_Send(&local_count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else {
        int total = local_count;
        for (int i = 1; i < numprocs; i++) {
            int count;
            MPI_Recv(&count, 1, MPI_INT, i, 0, MPI_COMM_WORLD, &status);
            total += count;
        }
        printf("Total primes less than %d: %d\n", N, total);
    }

    MPI_Finalize();
    return 0;
}
