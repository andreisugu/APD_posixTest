#include "mpi.h"
#include <stdio.h>

#define N 4  /* dimensiunea matricei N x N */

int main(int argc, char* argv[]) {
    int rank, numtasks, i, dest;
    float a[N * N];          /* matricea plata (row-major) */
    float b[2 * N];          /* buffer receptor: 2*N elemente (ambele diagonale) */
    MPI_Datatype diagtype;
    int blocklens[2 * N];
    int indices[2 * N];
    MPI_Status stat;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    /* -------------------------------------------------------
       Construim tipul MPI_Indexed care selecteaza ambele
       diagonale din matricea plata a[N*N].

       Matricea e stocata row-major: a[i][j] = a_flat[i*N + j]

       Diagonala principala:  a[i][i]     -> index flat = i*(N+1)
         ex N=4: 0, 5, 10, 15

       Diagonala secundara:   a[i][N-1-i] -> index flat = i*N + (N-1-i)
         ex N=4: 3, 6, 9, 12

       MPI_Type_indexed primeste un tablou de indici (pozitii)
       si un tablou de lungimi de bloc. Fiecare bloc are 1 element.
    ------------------------------------------------------- */
    for (i = 0; i < N; i++) {
        /* Primele N intrari = diagonala principala */
        indices[i]     = i * (N + 1);
        blocklens[i]   = 1;

        /* Urmatoarele N intrari = diagonala secundara */
        indices[N + i]   = i * N + (N - 1 - i);
        blocklens[N + i] = 1;
    }

    /* count = 2*N blocuri de cate 1 element MPI_FLOAT */
    MPI_Type_indexed(2 * N, blocklens, indices, MPI_FLOAT, &diagtype);
    // Pe scurt: indexed selecteaza 2*N elemente individuale din a[] 
    // conform indicilor din indices[] si le trateaza ca un bloc contiguu 
    // de 2*N elemente MPI_FLOAT. blocklens[] specifica ca fiecare bloc are 1 element,
    // deci se selecteaza exact 2*N elemente individuale.
    MPI_Type_commit(&diagtype);

    if (rank == 0) {
        /* Initializeaza matricea cu valorile 1 .. N*N */
        for (i = 0; i < N * N; i++)
            a[i] = (float)(i + 1);

        printf("Procesul 0 - Matricea %dx%d:\n", N, N);
        for (i = 0; i < N; i++) {
            for (int j = 0; j < N; j++)
                printf("%5.1f ", a[i * N + j]);
            printf("\n");
        }

        /* -------------------------------------------------------
           Trimite tipul indexed catre toate procesele.
           Send: 1 element de tip diagtype din a[]
                 -> MPI extrage automat cele 2*N valori selectate
                    si le trimite contiguu.
        ------------------------------------------------------- */
        for (dest = 1; dest < numtasks; dest++)
            MPI_Send(a, 1, diagtype, dest, 1, MPI_COMM_WORLD);

    } else {
        /* -------------------------------------------------------
           Recv: primeste 2*N float-uri contigue in b[].
           b[0..N-1]   = diagonala principala (in ordinea indicilor)
           b[N..2N-1]  = diagonala secundara
        ------------------------------------------------------- */
        MPI_Recv(b, 2 * N, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &stat);

        float suma = 0.0f;
        for (i = 0; i < 2 * N; i++)
            suma += b[i];

        printf("Procesul %d primeste: ", rank);
        for (i = 0; i < 2 * N; i++)
            printf("%.1f ", b[i]);
        printf("| Suma = %.1f\n", suma);
    }

    MPI_Type_free(&diagtype);
    MPI_Finalize();
    return 0;
}
