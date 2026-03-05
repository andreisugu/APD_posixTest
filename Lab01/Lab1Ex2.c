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

    int array_size = argc > 1 ? atoi(argv[1]) : 20;
    int search_element;
    int *array = (int *)malloc(array_size * sizeof(int));

    if (procid == 0) {
        srand(time(NULL));
        printf("Array: ");
        for (int i = 0; i < array_size; i++) {
            array[i] = rand() % 100;
            printf("%d ", array[i]);
        }
        printf("\n");
        search_element = argc > 2 ? atoi(argv[2]) : array[rand() % array_size];
        printf("Searching for: %d\n", search_element);
        /* Send array and search element to each worker */
        for (int i = 1; i < numprocs; i++) {
            MPI_Send(array, array_size, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&search_element, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(array, array_size, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&search_element, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
    }

    /* Each process searches its portion */
    int chunk = array_size / numprocs;
    int start = procid * chunk;
    int end = (procid == numprocs - 1) ? array_size : start + chunk;
    int local_pos = array_size; /* sentinel = not found */

    for (int i = start; i < end; i++)
        if (array[i] == search_element) { local_pos = i; break; }

    if (procid != 0) {
        MPI_Send(&local_pos, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    } else {
        int best = local_pos;
        for (int i = 1; i < numprocs; i++) {
            int pos;
            MPI_Recv(&pos, 1, MPI_INT, i, 2, MPI_COMM_WORLD, &status);
            if (pos < best) best = pos;
        }
        if (best < array_size)
            printf("Element %d found at position %d.\n", search_element, best);
        else
            printf("Not found.\n");
    }

    free(array);
    MPI_Finalize();
    return 0;
}
