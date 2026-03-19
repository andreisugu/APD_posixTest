/*
 * Exercitiul 1b: Cautarea unui element intr-un tablou
 * - Foloseste MPI_Scatter pentru a distribui array-ul
 * - Foloseste MPI_Gather pentru a colecta toate pozitiile gasite
 *
 * Compilare: mpicc -o ex1b ex1b_scatter_gather.c
 * Rulare:    mpirun -np 4 ./ex1b
 *
 * NOTA: ARRAY_SIZE trebuie sa fie divizibil cu numarul de procese.
 *       Schimba NP sau ARRAY_SIZE daca e necesar.
 */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define ARRAY_SIZE 20   /* Trebuie sa fie multiplu de NP */
#define TARGET 7        /* Elementul cautat */
#define NOT_FOUND -1    /* Marca pentru pozitie negasita */

int main(int argc, char *argv[]) {
    int rank, size;
    int array[ARRAY_SIZE];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int chunk = ARRAY_SIZE / size;  /* Elemente per proces */

    /* Procesul 0 initializeaza array-ul */
    if (rank == 0) {
        printf("=== Exercitiul 1b: MPI_Scatter + MPI_Gather ===\n");
        int init[] = {3, 7, 1, 9, 7, 2, 5, 7, 8, 4,
                      6, 7, 0, 7, 11, 3, 2, 7, 9, 1};
        for (int i = 0; i < ARRAY_SIZE; i++) {
            array[i] = init[i];
        }
        printf("Array: ");
        for (int i = 0; i < ARRAY_SIZE; i++) {
            printf("%d ", array[i]);
        }
        printf("\nCautam elementul: %d\n\n", TARGET);
    }

    /* --- Scatter: fiecare proces primeste 'chunk' elemente --- */
    int *local_array = (int *)malloc(chunk * sizeof(int));
    MPI_Scatter(array, chunk, MPI_INT,
                local_array, chunk, MPI_INT,
                0, MPI_COMM_WORLD);

    /*
     * Fiecare proces cauta TARGET in portiunea sa.
     * Stocam pozitiile gasite (indici GLOBALI) intr-un buffer local.
     * Buffer-ul are marimea 'chunk'; pozitiile negasite raman NOT_FOUND.
     */
    int *local_positions = (int *)malloc(chunk * sizeof(int));
    int found_count = 0;
    for (int i = 0; i < chunk; i++) {
        local_positions[i] = NOT_FOUND;
    }
    for (int i = 0; i < chunk; i++) {
        if (local_array[i] == TARGET) {
            int global_idx = rank * chunk + i;
            local_positions[found_count] = global_idx;
            found_count++;
        }
    }

    printf("Procesul %d: a gasit %d aparitii in intervalul [%d, %d)\n",
           rank, found_count, rank * chunk, rank * chunk + chunk);

    /* --- Gather: aduna pozitiile de la toti procesii in procesul 0 --- */
    int *all_positions = NULL;
    if (rank == 0) {
        all_positions = (int *)malloc(ARRAY_SIZE * sizeof(int));
    }

    MPI_Gather(local_positions, chunk, MPI_INT,
               all_positions, chunk, MPI_INT,
               0, MPI_COMM_WORLD);

    /* Procesul 0 afiseaza toate pozitiile gasite */
    if (rank == 0) {
        printf("\nToate pozitiile elementului %d:\n", TARGET);
        int total = 0;
        for (int i = 0; i < ARRAY_SIZE; i++) {
            if (all_positions[i] != NOT_FOUND) {
                printf("  Pozitia: %d  (valoare: %d)\n",
                       all_positions[i], array[all_positions[i]]);
                total++;
            }
        }
        if (total == 0) {
            printf("  Elementul NU a fost gasit.\n");
        } else {
            printf("  Total aparitii: %d\n", total);
        }
        free(all_positions);
    }

    free(local_array);
    free(local_positions);

    MPI_Finalize();
    return 0;
}
