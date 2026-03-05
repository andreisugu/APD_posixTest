/* Ex2: Search for an element in an array using M processes.
   Process 0 generates the array and distributes it; each process
   searches its chunk and reports the position back to process 0. */
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

    // Default array size is 20, but can be set via command line argument
    int array_size = 20;
    if (argc > 1) array_size = atoi(argv[1]);
    int search_element;
    int *array = (int *)malloc(array_size * sizeof(int));

    if (procid == 0) {
        /* Generate random array and pick search element */
        srand(time(NULL));
        printf("Array: ");
        for (int i = 0; i < array_size; i++) {
            array[i] = rand() % 100;
            printf("%d ", array[i]);
        }
        printf("\n");
        search_element = argc > 2 ? atoi(argv[2]) : array[rand() % array_size];
        printf("Searching for: %d\n", search_element);
        /* Send array and target to all workers */
        for (int i = 1; i < numprocs; i++) {
            MPI_Send(array, array_size, MPI_INT, i, 0, MPI_COMM_WORLD);
            MPI_Send(&search_element, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
        }
    } else {
        /* Workers receive the array and target from process 0 */
        MPI_Recv(array, array_size, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&search_element, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
    }

    /* Each process searches its chunk; local_pos = array_size means not found */
    int chunk = array_size / numprocs;
    int start = procid * chunk;
    int end = start + chunk;
    if (procid == numprocs - 1) {
        end = array_size;
    }
    int local_pos = array_size;

    for (int i = start; i < end; i++)
        if (array[i] == search_element) { local_pos = i; break; }

    if (procid != 0) {
        /* Workers send their result to process 0 */
        MPI_Send(&local_pos, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    } else {
        /* Process 0 collects results and picks the earliest position */
        /* To be brief: Process 0 receives the local positions from all workers
        and determines the best (smallest) position where the element was found.
        If none of the workers found the element, it will report "Not found."*/
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
