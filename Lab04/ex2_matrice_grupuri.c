/*
 * ============================================================
 * EXERCITIUL 2 - MPI Grupuri si Comunicatori
 * ============================================================
 * Problema: Se dau doua matrice A(M×K) si B(K×NN).
 * Impartiti procesele in doua grupuri:
 *   - Grupul 0: calculeaza jumatatea SUPERIOARA a matricei C = A×B
 *     (liniile 0 .. M/2 - 1)
 *   - Grupul 1: calculeaza jumatatea INFERIOARA a lui C
 *     (liniile M/2 .. M - 1)
 * La final, procesul 0 din MPI_COMM_WORLD colecteaza si
 * afiseaza matricea completa C.
 *
 * Design:
 *   - M=4 linii, K=3 coloane intermediare, NN=4 coloane finale
 *   - 4 procese: 2 per grup, fiecare proc calculeaza 1 linie din C
 *       proc 0 (grup 0, new_rank 0) -> linia 0 din C
 *       proc 1 (grup 0, new_rank 1) -> linia 1 din C
 *       proc 2 (grup 1, new_rank 0) -> linia 2 din C
 *       proc 3 (grup 1, new_rank 1) -> linia 3 din C
 *   - Procesul 0 aduna toate liniile cu MPI_Gather
 *
 * Compilare: mpicc -o ex2 ex2_matrice_grupuri.c
 * Rulare:    mpirun -np 4 ./ex2
 * ============================================================
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define M      4   /* linii in A si C                                    */
#define K      3   /* coloane in A = linii in B                          */
#define NN     4   /* coloane in B si C (NN evita conflict cu <limits.h>) */
#define NPROCS 4   /* 2 procese/grup × 2 grupuri                         */

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != NPROCS) {
        if (rank == 0)
            printf("Eroare: rulati cu exact %d procese!\n", NPROCS);
        MPI_Finalize();
        return 1;
    }

    /* Matricele A, B, C complete (fiecare proces va primi copii ale lor) */
    double A[M][K], B[K][NN], C[M][NN];

    /* ==========================================================
     * PASUL 1: Procesul 0 initializeaza A si B,
     * apoi le broadcast-eaza catre toate procesele.
     *
     * In practica, A si B ar putea fi citite din fisiere mari
     * sau generate de o etapa anterioara de calcul.
     * ========================================================== */
    if (rank == 0) {
        /* Initializare A: A[i][j] = i + j + 1 */
        printf("Matricea A (%dx%d):\n", M, K);
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < K; j++) {
                A[i][j] = i + j + 1.0;
                printf("  %5.1f", A[i][j]);
            }
            printf("\n");
        }

        /* Initializare B: matrice identitate extinsa (K×NN)
         * B[i][j] = 1 daca i==j, altfel 0
         * Asta inseamna C = A × B = A (primele K coloane) + 0 (restul) */
        printf("\nMatricea B (%dx%d) - identitate extinsa:\n", K, NN);
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < NN; j++) {
                B[i][j] = (i == j) ? 1.0 : 0.0;
                printf("  %5.1f", B[i][j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    /* Broadcast: trimite A si B de la procesul 0 la toate procesele.
     * Fiecare proces are nevoie de ambele matrice pentru a-si calcula
     * propria linie din C = A × B.
     * Trecem pointer la primul element si dimensiunea totala in elemente. */
    MPI_Bcast(A, M * K,  MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(B, K * NN, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* ==========================================================
     * PASUL 2: Definim si cream cele 2 grupuri de procese
     *
     * Grup 0: procesele {0, 1} -> jumatatea superioara a lui C
     * Grup 1: procesele {2, 3} -> jumatatea inferioara a lui C
     * ========================================================== */
    int procs_per_group = NPROCS / 2;    /* = 2 procese per grup          */
    int group_id = rank / procs_per_group; /* 0 sau 1                     */

    /* Rangurile globale ale proceselor din fiecare grup */
    int ranks_g0[2] = {0, 1};   /* jumatatea superioara */
    int ranks_g1[2] = {2, 3};   /* jumatatea inferioara */

    /* Extragem referinta la grupul global */
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    /* Cream subgrupul corespunzator fiecarui proces.
     * MPI_Group_incl construieste un nou grup selectand din world_group
     * procesele cu rangurile specificate in array.                       */
    MPI_Group my_group;
    if (group_id == 0)
        MPI_Group_incl(world_group, procs_per_group, ranks_g0, &my_group);
    else
        MPI_Group_incl(world_group, procs_per_group, ranks_g1, &my_group);

    /* Cream comunicatorul pentru grup.
     * COLECTIV: toate procesele din COMM_WORLD apeleaza,
     * chiar daca primesc un comunicator diferit.                         */
    MPI_Comm my_comm;
    MPI_Comm_create(MPI_COMM_WORLD, my_group, &my_comm);

    /* Rangul in cadrul noului grup (0 sau 1 in fiecare grup)             */
    int new_rank;
    MPI_Group_rank(my_group, &new_rank);

    /* ==========================================================
     * PASUL 3: Fiecare proces calculeaza linia sa din C
     *
     * Distributia liniilor:
     *   Grup 0, new_rank 0 (rank global 0) -> linia 0 din C
     *   Grup 0, new_rank 1 (rank global 1) -> linia 1 din C
     *   Grup 1, new_rank 0 (rank global 2) -> linia 2 din C
     *   Grup 1, new_rank 1 (rank global 3) -> linia 3 din C
     *
     * Fiecare proc calculeaza rows_per_proc linii consecutive
     * din jumatatea sa din C.
     * ========================================================== */
    int half_rows     = M / 2;           /* = 2 linii per grup            */
    int rows_per_proc = half_rows / procs_per_group; /* = 1 linie per proc */

    /* Linia de start din matricea C pentru acest proces:
     * - Grup 0, proc 0: start = 0*1 = 0
     * - Grup 0, proc 1: start = 1*1 = 1
     * - Grup 1, proc 0: start = 2 + 0*1 = 2
     * - Grup 1, proc 1: start = 2 + 1*1 = 3                             */
    int start_row = group_id * half_rows + new_rank * rows_per_proc;

    /* Calculul local al liniei(lor) din C = A × B
     * C[i][j] = suma peste k a A[i][k] * B[k][j]                        */
    double local_C[1][NN];  /* rows_per_proc = 1, deci un singur rand     */

    for (int i = 0; i < rows_per_proc; i++) {
        int row = start_row + i;       /* linia absoluta din C            */
        for (int j = 0; j < NN; j++) {
            local_C[i][j] = 0.0;
            for (int p = 0; p < K; p++) {
                /* Inmultire element cu element si acumulare              */
                local_C[i][j] += A[row][p] * B[p][j];
            }
        }
        printf("[Grup %d, proc %d] Calculez linia %d din C\n",
               group_id, rank, row);
    }

    /* ==========================================================
     * PASUL 4: Colectarea liniilor la procesul 0 cu MPI_Gather
     *
     * MPI_Gather aduna de la fiecare proces rows_per_proc*NN
     * double-uri si le concateneaza la procesul radacina (0).
     * Ordinea: mai intai datele proc 0, apoi proc 1, etc.
     *
     * Rezultat la proc 0: C_flat = [linia0 | linia1 | linia2 | linia3]
     * ========================================================== */
    double C_flat[M * NN];  /* buffer complet, alocat doar la proc 0     */

    /* Toți procesele trimit rows_per_proc*NN = 1*4 = 4 double-uri
     * Procesul 0 primeste NPROCS * 4 = 16 double-uri (matricea completa) */
    MPI_Gather(local_C,              /* buffer de trimitere (local)       */
               rows_per_proc * NN,   /* = 4 double-uri trimise de fiecare */
               MPI_DOUBLE,
               C_flat,               /* buffer de receptie (doar la proc 0)*/
               rows_per_proc * NN,   /* = 4 double-uri primite de la fiecare*/
               MPI_DOUBLE,
               0,                    /* procesul radacina                 */
               MPI_COMM_WORLD);

    /* ==========================================================
     * PASUL 5: Procesul 0 reconstruieste si afiseaza C
     * ========================================================== */
    if (rank == 0) {
        /* Reconstruim C din forma liniara (row-major) */
        for (int i = 0; i < M; i++)
            for (int j = 0; j < NN; j++)
                C[i][j] = C_flat[i * NN + j];

        printf("\nMatricea C = A x B (%dx%d):\n", M, NN);
        for (int i = 0; i < M; i++) {
            printf("  Linia %d [calculata de Grupul %d]: ", i, i < M/2 ? 0 : 1);
            for (int j = 0; j < NN; j++)
                printf("  %5.1f", C[i][j]);
            printf("\n");
        }
    }

    /* ==========================================================
     * PASUL 6: Eliberarea resurselor
     *
     * Eliberam in ordinea: comunicator -> subgrup -> grup global
     * ========================================================== */
    MPI_Comm_free(&my_comm);
    MPI_Group_free(&my_group);
    MPI_Group_free(&world_group);

    MPI_Finalize();
    return 0;
}
