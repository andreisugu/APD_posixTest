/* Ex3: Each process generates m random numbers (<=1000), prints them,
   computes their sum, then prints the time it took to complete. */
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    int numprocs, procid;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);

    // Default m=100, but can be set via command line argument; enforce m>=100
    int m = 100;
    if (argc > 1) {
        m = atoi(argv[1]);
        if (m < 100) {
            m = 100;
        }
    }

    double start = MPI_Wtime(); // start timer
    srand(time(NULL) + procid * 1000); /* random seed per process */

    long long sum = 0;
    printf("Process %d numbers: ", procid);
    // Generate m random numbers, print them, and compute their sum
    for (int i = 0; i < m; i++) {
        int val = rand() % 1001;
        printf("%d ", val);
        sum += val;
    }
    printf("\nProcess %d: sum = %lld\n", procid, sum);

    /* Stop timer and print elapsed time for this process */
    double elapsed = MPI_Wtime() - start;
    printf("Process %d: time = %f s\n", procid, elapsed);

    MPI_Finalize();
    return 0;
}
