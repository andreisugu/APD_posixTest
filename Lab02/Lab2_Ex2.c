/*
 * Exercitiul 2 - Laborator 2 MPI
 * Adunarea elementelor a doua matrici cu numere in virgula mobila.
 * - Primeste doua matrici de intrare de dimensiuni egale (ROWS x COLS).
 * - Produce o matrice de iesire C = A + B (suma element cu element).
 * - Distribuie randurile prin MPI_Scatter, aduna rezultatele cu MPI_Gather.
 * - Rezultatul final este stocat in matricea C.
 *
 * Nota: ROWS trebuie sa fie divizibil cu numarul de procese.
 *
 * Compilare: mpicc Lab2_Ex2.c -o Lab2_Ex2
 * Rulare:    mpirun -np 4 ./Lab2_Ex2
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ROWS 4
#define COLS 5

static void print_matrix(const char *name, float *M, int rows, int cols) {
    printf("Matricea %s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6.1f ", M[i * cols + j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[]) {
    int rank, size;

    /* Matricile A, B, C sunt alocate doar pe procesul 0 */
    float *A = NULL, *B = NULL, *C = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (ROWS % size != 0) {
        if (rank == 0)
            fprintf(stderr, "Eroare: ROWS=%d nu este divizibil cu np=%d\n", ROWS, size);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rows_per_proc = ROWS / size;
    int chunk         = rows_per_proc * COLS;   /* elemente per proces */

    if (rank == 0) {
        A = (float *)malloc(ROWS * COLS * sizeof(float));
        B = (float *)malloc(ROWS * COLS * sizeof(float));
        C = (float *)malloc(ROWS * COLS * sizeof(float));

        /* A[i] = i+1, B[i] = ROWS*COLS - i => C[i] = ROWS*COLS + 1 (constant) */
        for (int i = 0; i < ROWS * COLS; i++) {
            A[i] = (float)(i + 1);
            B[i] = (float)(ROWS * COLS - i);
        }
        print_matrix("A", A, ROWS, COLS);
        printf("\n");
        print_matrix("B", B, ROWS, COLS);
        printf("\n");
    }

    /* Buffere locale pentru fiecare proces */
    float *local_A = (float *)malloc(chunk * sizeof(float));
    float *local_B = (float *)malloc(chunk * sizeof(float));
    float *local_C = (float *)malloc(chunk * sizeof(float));

    /* Distribuim randurile din A si B (sendbuf = NULL pe non-root e valid MPI) */
    MPI_Scatter(A, chunk, MPI_FLOAT, local_A, chunk, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, chunk, MPI_FLOAT, local_B, chunk, MPI_FLOAT, 0, MPI_COMM_WORLD);

    /* Calculam suma locala */
    for (int i = 0; i < chunk; i++) {
        local_C[i] = local_A[i] + local_B[i];
    }

    /* Adunam rezultatele in C */
    MPI_Gather(local_C, chunk, MPI_FLOAT, C, chunk, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        print_matrix("C = A + B", C, ROWS, COLS);

        /* Verificare: fiecare element C[i] trebuie sa fie ROWS*COLS + 1 */
        float expected = (float)(ROWS * COLS + 1);
        int ok = 1;
        for (int i = 0; i < ROWS * COLS; i++) {
            if (C[i] != expected) { ok = 0; break; }
        }
        printf("\nVerificare: toate elementele C ar trebui sa fie %.1f => %s\n",
               expected, ok ? "CORECT" : "INCORECT");

        free(A); free(B); free(C);
    }

    free(local_A);
    free(local_B);
    free(local_C);
    MPI_Finalize();
    return 0;
}
