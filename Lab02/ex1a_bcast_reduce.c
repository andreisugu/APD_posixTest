/*
 * Exercitiul 1a: Cautarea unui element intr-un tablou
 * - Foloseste MPI_Bcast pentru a trimite array-ul
 * - Gaseste indexul MAXIM al pozitiei cu MPI_Reduce
 *
 * Compilare: mpicc -o ex1a ex1a_bcast_reduce.c
 * Rulare:    mpirun -np 4 ./ex1a
 */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define ARRAY_SIZE 20
#define TARGET 7   /* Elementul cautat */

int main(int argc, char *argv[]) {
    int rank, size;
    int array[ARRAY_SIZE];
    int local_max_index = -1;  /* -1 = nefound */
    int global_max_index = -1;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Procesul 0 initializeaza array-ul */
    if (rank == 0) {
        printf("=== Exercitiul 1a: MPI_Bcast + MPI_Reduce ===\n");
        /* Array exemplu cu duplicare pentru TARGET */
        int init[] = {3, 7, 1, 9, 7, 2, 5, 7, 8, 4,
                      6, 7, 0, 7, 11, 3, 2, 7, 9, 1};
        for (int i = 0; i < ARRAY_SIZE; i++) {
            array[i] = init[i];
        }
        printf("Array: ");
        for (int i = 0; i < ARRAY_SIZE; i++) {
            printf("%d ", array[i]);
        }
        printf("\nCautam elementul: %d\n", TARGET);
    }

    /* Broadcast array-ul catre totate procesele */
    MPI_Bcast(array, ARRAY_SIZE, MPI_INT, 0, MPI_COMM_WORLD);

    /* Fiecare proces cauta in bucata sa din array */
    int chunk = ARRAY_SIZE / size;
    int start = rank * chunk;
    int end   = (rank == size - 1) ? ARRAY_SIZE : start + chunk;

    for (int i = start; i < end; i++) {
        if (array[i] == TARGET) {
            if (i > local_max_index) {
                local_max_index = i;
            }
        }
    }

    /* MPI_Reduce pentru a gasi indexul maxim global */
    MPI_Reduce(&local_max_index, &global_max_index, 1,
               MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (global_max_index == -1) {
            printf("Elementul %d NU a fost gasit in array.\n", TARGET);
        } else {
            printf("Indexul maxim al elementului %d este: %d\n",
                   TARGET, global_max_index);
        }
    }

    MPI_Finalize();
    return 0;
}
