#include "mpi.h"
#include <stdio.h>

#define ROWS 4
#define COLS 8  /* COLS >= 2*(numtasks-1), ex: 4 procese receptor -> COLS=8 */

int main(int argc, char* argv[]) {
    int rank, numtasks, i, dest;
    float a[ROWS][COLS];
    float b[ROWS * 2];   /* buffer receptor: ROWS randuri x 2 coloane */
    MPI_Datatype colpairtype;
    MPI_Status stat;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    /* -------------------------------------------------------
       Construim tipul MPI_Vector care descrie 2 coloane
       consecutive dintr-o matrice row-major ROWS x COLS.

       MPI_Type_vector(count, blocklength, stride, old, new):
         count       = ROWS   -> cate blocuri (un bloc per rand)
         blocklength = 2      -> 2 elemente consecutive per rand
                                 (coloanele c si c+1 din randul curent)
         stride      = COLS   -> distanta (in elemente) intre
                                 inceputul a doua blocuri consecutive
                                 (= latimea unui rand)

       Vizual pentru ROWS=4, COLS=8, coloanele 2-3:
         rand 0: [_, _, X, X, _, _, _, _]
         rand 1: [_, _, X, X, _, _, _, _]
         rand 2: [_, _, X, X, _, _, _, _]
         rand 3: [_, _, X, X, _, _, _, _]
    ------------------------------------------------------- */
    MPI_Type_vector(ROWS, 2, COLS, MPI_FLOAT, &colpairtype);
    MPI_Type_commit(&colpairtype);

    if (rank == 0) {
        /* Initializeaza matricea */
        int j;
        for (i = 0; i < ROWS; i++)
            for (j = 0; j < COLS; j++)
                a[i][j] = (float)(i * COLS + j + 1);

        printf("Procesul 0 - Matricea %dx%d initiala:\n", ROWS, COLS);
        for (i = 0; i < ROWS; i++) {
            for (j = 0; j < COLS; j++)
                printf("%5.1f ", a[i][j]);
            printf("\n");
        }

        /* -------------------------------------------------------
           Trimite fiecarui proces cate 2 coloane consecutive.
           Procesul dest primeste coloanele: 2*(dest-1) si 2*(dest-1)+1

           &a[0][col_start] = adresa de start a primei celule
           din cele 2 coloane ce trebuie trimise.
           MPI foloseste tipul vector sa sara corect intre randuri.
        ------------------------------------------------------- */
        for (dest = 1; dest < numtasks; dest++) {
            int col_start = (dest - 1) * 2;
            MPI_Send(&a[0][col_start], 1, colpairtype, dest, 1, MPI_COMM_WORLD);
        }

    } else {
        /* -------------------------------------------------------
           Recv: primeste ROWS*2 float-uri contigue in b[].
           Datele sosesc rand cu rand:
             b[0], b[1]         = randul 0, col c si c+1
             b[2], b[3]         = randul 1, col c si c+1
             ...
        ------------------------------------------------------- */
        MPI_Recv(b, ROWS * 2, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &stat);

        float suma = 0.0f, maxval = b[0], medie;
        for (i = 0; i < ROWS * 2; i++) {
            suma += b[i];
            if (b[i] > maxval) maxval = b[i];
        }
        medie = suma / (ROWS * 2);

        printf("Procesul %d (col %d-%d): ", rank, (rank-1)*2, (rank-1)*2+1);
        for (i = 0; i < ROWS * 2; i++)
            printf("%.1f ", b[i]);
        printf("| Suma=%.1f  Media=%.2f  Max=%.1f\n", suma, medie, maxval);
    }

    MPI_Type_free(&colpairtype);
    MPI_Finalize();
    return 0;
}
