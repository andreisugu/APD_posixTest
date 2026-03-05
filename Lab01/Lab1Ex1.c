/* Ex1: Print all prime numbers less than N using M processes.
   Each process gets a block of numbers to check independently. */
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>

int is_prime(int n) {
    if (n < 2) return 0;
    for (int d = 2; d * d <= n; d++)
        if (n % d == 0) return 0;
    return 1;
}

int main(int argc, char *argv[]) {
    int numprocs, procid;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);

    // Default N=100, but can be set via command line argument
    int N = 100;
    if (argc > 1) N = atoi(argv[1]);

    /* Divide [2, N) into blocks, one per process */
    int range = N / numprocs;
    int start = procid * range;
    int end = (procid + 1) * range;
    if (procid == numprocs - 1) {
        end = N;
    }
    // This does this: Process 0 checks [0, range), 
    // Process 1 checks [range, 2*range), ..., Process M-1 checks [(M-1)*range, N)
    if (procid == 0) start = 2; /* skip 0 and 1 */

    /* Each process checks and prints its own primes */
    for (int i = start; i < end; i++)
        if (is_prime(i))
            printf("[Process %d] Prime: %d\n", procid, i);

    MPI_Finalize();
    return 0;
}
