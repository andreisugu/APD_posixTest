/*
 * Exercitiul 1a - Laborator 2 MPI
 * Cauta un element intr-un tablou.
 * - Foloseste MPI_Bcast pentru a trimite array-ul.
 * - Daca elementul este gasit, afiseaza indexul maxim al pozitiei.
 * - Foloseste MPI_Reduce (MPI_MAX) pentru a calcula pozitia maxima.
 *
 * Compilare: mpicc Lab2_Ex1a.c -o Lab2_Ex1a
 * Rulare:    mpirun -np 4 ./Lab2_Ex1a
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 20

int main(int argc, char *argv[]) {
    int rank, size;
    int arr[N];
    int target;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        /* Array cu mai multe aparitii ale lui 7, pentru un test mai relevant */
        int init[] = {3, 7, 1, 9, 7, 2, 5, 7, 8, 4,
                      6, 7, 0, 7, 11, 3, 2, 7, 9, 1};
        for (int i = 0; i < N; i++) arr[i] = init[i];
        target = 7;

        printf("Array (%d elemente): ", N);
        for (int i = 0; i < N; i++) printf("%d ", arr[i]);
        printf("\nElement cautat: %d\n", target);
    }

    /* Trimitem array-ul si targetul la toate procesele */
    MPI_Bcast(arr,    N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&target, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Fiecare proces cauta in propria portiune */
    int chunk = N / size;
    int start = rank * chunk;
    int end   = (rank == size - 1) ? N : start + chunk;

    int local_max_pos = -1;
    for (int i = start; i < end; i++) {
        if (arr[i] == target && i > local_max_pos) {
            local_max_pos = i;
        }
    }

    /* Reducem pentru a gasi pozitia maxima globala */
    int global_max_pos = -1;
    MPI_Reduce(&local_max_pos, &global_max_pos, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (global_max_pos == -1) {
            printf("Elementul %d NU a fost gasit in array.\n", target);
        } else {
            printf("Pozitia maxima a elementului %d: %d\n", target, global_max_pos);
        }
    }

    MPI_Finalize();
    return 0;
}
