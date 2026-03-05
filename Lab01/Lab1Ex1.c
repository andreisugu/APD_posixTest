/* Ex1: Print all prime numbers less than N using M processes.
   Each process gets a block of numbers to check independently. */
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

/* Returns 1 if n is prime, 0 otherwise */
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

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);

    int N = argc > 1 ? atoi(argv[1]) : 100;

    /* Divide [2, N) into blocks, one per process */
    int range = N / numprocs;
    int start = procid * range;
    int end = (procid == numprocs - 1) ? N : (procid + 1) * range;
    if (procid == 0) start = 2; /* skip 0 and 1 */

    /* Each process checks and prints its own primes */
    for (int i = start; i < end; i++)
        if (is_prime(i))
            printf("[Process %d] Prime: %d\n", procid, i);

    MPI_Finalize();
    return 0;
}
