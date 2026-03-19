/*
 * Exercitiul 1b - Laborator 2 MPI
 * Cauta un element intr-un tablou.
 * - Foloseste MPI_Scatter pentru a distribui array-ul.
 * - Daca elementul este gasit de mai multe ori, tipareste TOATE pozitiile.
 * - Foloseste MPI_Gather pentru a aduna pozitiile inapoi la root.
 *
 * Nota: N trebuie sa fie divizibil cu numarul de procese.
 *
 * Compilare: mpicc Lab2_Ex1b.c -o Lab2_Ex1b
 * Rulare:    mpirun -np 4 ./Lab2_Ex1b
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

    /* Verificam ca N este divizibil cu size */
    if (N % size != 0) {
        if (rank == 0)
            fprintf(stderr, "Eroare: N=%d nu este divizibil cu np=%d\n", N, size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int chunk = N / size;

    if (rank == 0) {
        /* Array cu mai multe aparitii ale lui 7, pentru un test mai relevant */
        int init[] = {3, 7, 1, 9, 7, 2, 5, 7, 8, 4,
                      6, 7, 0, 7, 11, 3, 2, 7, 9, 1};
        for (int i = 0; i < N; i++) arr[i] = init[i];
        target = 7;

        printf("Array (%d elemente): ", N);
        for (int i = 0; i < N; i++) printf("%d ", arr[i]);
        printf("\nElement cautat: %d\n\n", target);
    }

    /* Distribuim targetul si portiunile din array */
    MPI_Bcast(&target, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int *local_arr = (int *)malloc(chunk * sizeof(int));
    MPI_Scatter(arr, chunk, MPI_INT, local_arr, chunk, MPI_INT, 0, MPI_COMM_WORLD);

    /*
     * Fiecare proces construieste un vector local_result[i]:
     *  - indicele global daca arr[i] == target
     *  - -1 altfel
     */
    int *local_result = (int *)malloc(chunk * sizeof(int));
    int found_count = 0;
    for (int i = 0; i < chunk; i++) local_result[i] = -1;
    for (int i = 0; i < chunk; i++) {
        if (local_arr[i] == target) {
            local_result[found_count++] = rank * chunk + i;
        }
    }
    printf("Procesul %d: a gasit %d aparitii in intervalul [%d, %d)\n",
           rank, found_count, rank * chunk, rank * chunk + chunk);

    /* Adunam toate rezultatele inapoi la root */
    int *all_results = NULL;
    if (rank == 0) {
        all_results = (int *)malloc(N * sizeof(int));
    }
    MPI_Gather(local_result, chunk, MPI_INT, all_results, chunk, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\nToate pozitiile elementului %d:\n", target);
        int found = 0;
        for (int i = 0; i < N; i++) {
            if (all_results[i] != -1) {
                printf("  Pozitia: %d  (valoare: %d)\n",
                       all_results[i], arr[all_results[i]]);
                found++;
            }
        }
        if (found == 0) printf("  (nu a fost gasit)\n");
        else printf("  Total aparitii: %d\n", found);
        free(all_results);
    }

    free(local_arr);
    free(local_result);
    MPI_Finalize();
    return 0;
}
