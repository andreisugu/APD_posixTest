/*
 * ============================================================
 * TEMA LABORATOR: Roy-Floyd-Warshall (MPI) - SPECIFIC MATRIX
 * ============================================================
 * Problema: Se da o matrice de adiacenta D SPECIFICATA
 * reprezentand distantele dintr-un graf. Sa se gaseasca
 * distantele minime intre oricare doua noduri.
 *
 * Matricea (din imagine):
 *     1  2  3  4
 *  1  0  3  ∞  7
 *  2  8  0  2  ∞
 *  3  5  ∞  0  1
 *  4  2  ∞  ∞  0
 *
 * Strategie de paralelizare:
 * - N=4 noduri, 4 procese.
 * - Matricea D este impartita pe linii (row-wise).
 * - Fiecare proces va detine N/NPROCS = 1 linie din matrice.
 * - La fiecare pas 'k' (din cele N):
 * 1. Procesul care detine linia 'k' ii face broadcast (MPI_Bcast)
 * catre toate celelalte procese.
 * 2. Fiecare proces isi actualizeaza bucata sa de matrice
 * folosind linia 'k' primita.
 * - La final, procesul 0 aduna liniile procesate cu MPI_Gather
 * si afiseaza timpul de executie.
 *
 * Compilare: mpicc -o main_specific main_specific.c
 * Rulare:    mpirun -np 4 ./main_specific
 * ============================================================
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 4          /* Numarul de noduri (linii/coloane)       */
#define NPROCS 4     /* Numarul asteptat de procese MPI         */
#define INF 99999    /* Reprezentarea infinitului pentru graf   */

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Verificam numarul corect de procese */
    if (size != NPROCS)
    {
        if (rank == 0)
            printf("Eroare: rulati cu exact %d procese!\n", NPROCS);
        MPI_Finalize();
        return 1;
    }

    int rows_per_proc = N / NPROCS; /* = 1 linie per proces */

    /* Matricea globala D, alocata complet doar pe procesul radacina (0) */
    int *D = NULL;

    /* Bucata locala a matricei pentru fiecare proces (vector 1D plat) */
    int *local_D = (int *)malloc(rows_per_proc * N * sizeof(int));
    
    /* Buffer pentru a stoca linia 'k' la momentul broadcast-ului */
    int *row_k = (int *)malloc(N * sizeof(int));

    /* ==========================================================
     * PASUL 1: Initializarea datelor la procesul 0
     * ========================================================== */
    if (rank == 0)
    {
        D = (int *)malloc(N * N * sizeof(int));

        /* Matricea din imagine:
         *     1  2  3  4
         *  1  0  3  ∞  7
         *  2  8  0  2  ∞
         *  3  5  ∞  0  1
         *  4  2  ∞  ∞  0
         */
        int matrix_init[4][4] = {
            {0,   3, INF,  7},
            {8,   0,   2, INF},
            {5, INF,   0,   1},
            {2, INF, INF,   0}
        };

        printf("Matricea initiala a distantelor (din imagine, INF=%d):\n\n", INF);
        printf("     1     2     3     4\n");
        for (int i = 0; i < N; i++)
        {
            printf("%d ", i + 1);
            for (int j = 0; j < N; j++)
            {
                D[i * N + j] = matrix_init[i][j];
                
                if (D[i * N + j] == INF)
                    printf("  INF");
                else
                    printf(" %5d", D[i * N + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    /* Asiguram sincronizarea inainte de a porni cronometrul */
    MPI_Barrier(MPI_COMM_WORLD);

    /* ==========================================================
     * PASUL 2: Masurarea timpului (START)
     * ========================================================== */
    double start_time = MPI_Wtime();

    /* ==========================================================
     * PASUL 3: Distribuirea liniilor matricei catre procese
     * Folosim MPI_Scatter pentru a trimite din D cate 
     * (rows_per_proc * N) elemente catre fiecare local_D.
     * ========================================================== */
    MPI_Scatter(D, rows_per_proc * N, MPI_INT,
                local_D, rows_per_proc * N, MPI_INT,
                0, MPI_COMM_WORLD);

    /* ==========================================================
     * PASUL 4: Algoritmul Roy-Floyd-Warshall (Bucla k)
     * ========================================================== */
    for (int k = 0; k < N; k++)
    {
        /* Determinam care proces detine linia k in memoria sa locala */
        int owner_rank = k / rows_per_proc;

        /* Daca eu sunt procesul "proprietar", extrag linia k in bufferul row_k */
        if (rank == owner_rank)
        {
            int local_k_index = k % rows_per_proc; /* indexul relativ in local_D */
            for (int j = 0; j < N; j++)
            {
                row_k[j] = local_D[local_k_index * N + j];
            }
        }

        /* Proprietarul face broadcast la linia k catre toata lumea */
        MPI_Bcast(row_k, N, MPI_INT, owner_rank, MPI_COMM_WORLD);

        /* Fiecare proces isi actualizeaza portiunea de matrice */
        for (int i = 0; i < rows_per_proc; i++)
        {
            for (int j = 0; j < N; j++)
            {
                int dist_ik = local_D[i * N + k];
                int dist_kj = row_k[j];

                /* Evitam overflow la adunarea cu infinit */
                if (dist_ik != INF && dist_kj != INF)
                {
                    if (dist_ik + dist_kj < local_D[i * N + j])
                    {
                        local_D[i * N + j] = dist_ik + dist_kj;
                    }
                }
            }
        }
    }

    /* ==========================================================
     * PASUL 5: Colectarea matricei finale
     * Folosim MPI_Gather pentru a aduce bucatile locale inapoi
     * in matricea globala D de pe procesul 0.
     * ========================================================== */
    MPI_Gather(local_D, rows_per_proc * N, MPI_INT,
               D, rows_per_proc * N, MPI_INT,
               0, MPI_COMM_WORLD);

    /* ==========================================================
     * PASUL 6: Masurarea timpului (STOP)
     * ========================================================== */
    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    /* ==========================================================
     * PASUL 7: Afisare rezultate la procesul radacina
     * ========================================================== */
    if (rank == 0)
    {
        printf("Matricea finala a distantelor minime:\n\n");
        printf("     1     2     3     4\n");
        for (int i = 0; i < N; i++)
        {
            printf("%d ", i + 1);
            for (int j = 0; j < N; j++)
            {
                if (D[i * N + j] == INF)
                    printf("  INF");
                else
                    printf(" %5d", D[i * N + j]);
            }
            printf("\n");
        }

        printf("\n=== TIMP DE EXECUTIE ===\n");
        printf("Timp MPI Roy-Floyd (specific): %f secunde\n", end_time - start_time);
        
        free(D);
    }

    /* Eliberarea memoriei locale */
    free(local_D);
    free(row_k);

    MPI_Finalize();
    return 0;
}
