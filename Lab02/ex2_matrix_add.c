/*
 * Exercitiul 2: Adunarea element cu element a doua matrici float
 * - MPI_Scatter  : distribuie liniile matricilor A si B catre procese
 * - MPI_Gather   : aduna liniile rezultat inapoi in procesul 0
 * - Rezultatul final este stocat in matricea C (pe procesul 0)
 *
 * Compilare: mpicc -o ex2 ex2_matrix_add.c -lm
 * Rulare:    mpirun -np 4 ./ex2
 *
 * NOTA: ROWS trebuie sa fie multiplu de numarul de procese.
 */

#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

#define ROWS 4   /* Numar de linii  (multiplu de NP) */
#define COLS 5   /* Numar de coloane */

/* Afiseaza o matrice row-major stocata liniar */
void print_matrix(const char *name, float *M, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        printf("  [ ");
        for (int j = 0; j < cols; j++) {
            printf("%6.1f ", M[i * cols + j]);
        }
        printf("]\n");
    }
}

int main(int argc, char *argv[]) {
    int rank, size;

    /* Matricile sunt alocate liniar (row-major) */
    float *A = NULL, *B = NULL, *C = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows_per_proc = ROWS / size;           /* Linii per proces   */
    int elems_per_proc = rows_per_proc * COLS; /* Elemente per proces */

    /* Procesul 0 aloca si initializeaza matricile */
    if (rank == 0) {
        A = (float *)malloc(ROWS * COLS * sizeof(float));
        B = (float *)malloc(ROWS * COLS * sizeof(float));
        C = (float *)malloc(ROWS * COLS * sizeof(float));

        printf("=== Exercitiul 2: Adunare matrice cu MPI ===\n\n");

        /* Initializare A[i][j] = i*COLS + j + 1  (1,2,3,...) */
        for (int i = 0; i < ROWS * COLS; i++) {
            A[i] = (float)(i + 1);
            B[i] = (float)((ROWS * COLS) - i); /* B descrescator     */
        }

        print_matrix("Matricea A", A, ROWS, COLS);
        printf("\n");
        print_matrix("Matricea B", B, ROWS, COLS);
        printf("\n");
    }

    /* Buffer-e locale pentru bucatile din A, B, C */
    float *local_A = (float *)malloc(elems_per_proc * sizeof(float));
    float *local_B = (float *)malloc(elems_per_proc * sizeof(float));
    float *local_C = (float *)malloc(elems_per_proc * sizeof(float));

    /* --- Scatter: distribuie linii din A si B --- */
    MPI_Scatter(A, elems_per_proc, MPI_FLOAT,
                local_A, elems_per_proc, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    MPI_Scatter(B, elems_per_proc, MPI_FLOAT,
                local_B, elems_per_proc, MPI_FLOAT,
                0, MPI_COMM_WORLD);

    /* --- Calcul local: C[i] = A[i] + B[i] --- */
    for (int i = 0; i < elems_per_proc; i++) {
        local_C[i] = local_A[i] + local_B[i];
    }

    /* --- Gather: aduna liniile rezultat in C pe procesul 0 --- */
    MPI_Gather(local_C, elems_per_proc, MPI_FLOAT,
               C, elems_per_proc, MPI_FLOAT,
               0, MPI_COMM_WORLD);

    /* Procesul 0 afiseaza rezultatul */
    if (rank == 0) {
        print_matrix("Matricea C = A + B", C, ROWS, COLS);

        /* Verificare: fiecare element C[i] trebuie sa fie ROWS*COLS+1 */
        printf("\nVerificare: toate elementele C ar trebui sa fie %.1f\n",
               (float)(ROWS * COLS + 1));
        int ok = 1;
        for (int i = 0; i < ROWS * COLS; i++) {
            if (C[i] != (float)(ROWS * COLS + 1)) {
                ok = 0;
                break;
            }
        }
        printf("Rezultat: %s\n", ok ? "CORECT" : "INCORECT");

        free(A); free(B); free(C);
    }

    free(local_A); free(local_B); free(local_C);

    MPI_Finalize();
    return 0;
}
