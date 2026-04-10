/*
 * ============================================================
 * EXERCITIUL 1 - MPI Grupuri si Comunicatori
 * ============================================================
 * Problema: Se da un vector cu N elemente. Sa se calculeze
 * suma, produsul, minimul si maximul elementelor, SIMULTAN,
 * folosind 4 grupuri de procese.
 *
 * Strategie:
 *   - Vectorul de N elemente este distribuit uniform intre
 *     toate cele 8 procese (cate 2 elemente per proces).
 *   - Procesele sunt impartite in 4 grupuri de cate 2:
 *       Grup 0 (proc 0,1) -> calculeaza SUMA
 *       Grup 1 (proc 2,3) -> calculeaza PRODUSUL
 *       Grup 2 (proc 4,5) -> calculeaza MINIMUL
 *       Grup 3 (proc 6,7) -> calculeaza MAXIMUL
 *   - Cele 4 operatii ruleaza simultan pe comunicatori diferiti.
 *   - La final, liderii grupurilor trimit rezultatele
 *     partiale catre procesul 0, care calculeaza si afiseaza
 *     rezultatele globale finale.
 *
 * Compilare: mpicc -o ex1 ex1_grupe_operatii.c
 * Rulare:    mpirun -np 8 ./ex1
 * ============================================================
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define NPROCS 8   /* Numar fix de procese (2 per grup × 4 grupuri)    */
#define N      16  /* Marimea vectorului (multiplu de NPROCS)           */

int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* Verificam numarul corect de procese */
    if (size != NPROCS) {
        if (rank == 0)
            printf("Eroare: rulati cu exact %d procese!\n", NPROCS);
        MPI_Finalize();
        return 1;
    }

    /* ==========================================================
     * PASUL 1: Initializarea datelor locale
     *
     * Vectorul global: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
     * Fiecare proces detine N/NPROCS = 2 elemente consecutive:
     *   proc 0 -> {1, 2}   | proc 1 -> {3, 4}   | proc 2 -> {5, 6}
     *   proc 3 -> {7, 8}   | proc 4 -> {9, 10}  | proc 5 -> {11, 12}
     *   proc 6 -> {13, 14} | proc 7 -> {15, 16}
     * ========================================================== */
    int chunk = N / NPROCS;             /* = 2 elemente per proces          */
    int *local_data = malloc(chunk * sizeof(int));

    for (int i = 0; i < chunk; i++) {
        /* Elementul i al procesului rank este: rank*chunk + i + 1         */
        local_data[i] = rank * chunk + i + 1;
    }

    /* ==========================================================
     * PASUL 2: Calcul valori partiale locale
     *
     * Fiecare proces isi calculeaza local suma/produsul/min/max
     * pentru propriile elemente. Aceste valori vor fi combinate
     * ulterior prin operatii colective in cadrul grupului.
     * ========================================================== */
    long long local_sum  = 0;
    long long local_prod = 1;
    int       local_min  = local_data[0];
    int       local_max  = local_data[0];

    for (int i = 0; i < chunk; i++) {
        local_sum  += local_data[i];
        local_prod *= local_data[i];
        if (local_data[i] < local_min) local_min = local_data[i];
        if (local_data[i] > local_max) local_max = local_data[i];
    }

    /* ==========================================================
     * PASUL 3: Definirea si crearea celor 4 grupuri MPI
     *
     * Impartim cele 8 procese in 4 grupuri de 2:
     *   group_id = rank / 2
     *   Grup 0: proc {0,1}  -> SUMA
     *   Grup 1: proc {2,3}  -> PRODUS
     *   Grup 2: proc {4,5}  -> MINIM
     *   Grup 3: proc {6,7}  -> MAXIM
     * ========================================================== */
    int procs_per_group = 2;
    int group_id = rank / procs_per_group;  /* 0, 1, 2 sau 3               */

    /* Array cu rangurile proceselor din fiecare grup.
     * Grupul i contine procesele: {i*2, i*2+1}                            */
    int group_ranks[4][2] = {
        {0, 1},   /* Grup 0: SUMA     */
        {2, 3},   /* Grup 1: PRODUS   */
        {4, 5},   /* Grup 2: MINIM    */
        {6, 7}    /* Grup 3: MAXIM    */
    };

    /* Extragem grupul global asociat lui MPI_COMM_WORLD.
     * Un grup MPI este o lista ordonata de procese; nu poate fi
     * folosit direct pentru comunicare, dar serveste ca baza
     * pentru derivarea de subgrupuri.                                      */
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    /* Cream subgrupul pentru procesul curent.
     * MPI_Group_incl selecteaza din world_group procesele
     * ale caror ranguri se afla in array-ul dat (procs_per_group elemente).
     * Rezultatul este stocat in my_group.                                  */
    MPI_Group my_group;
    MPI_Group_incl(world_group, procs_per_group,
                   group_ranks[group_id], &my_group);

    /* Cream comunicatorul corespunzator noului grup.
     *
     * IMPORTANT: MPI_Comm_create este COLECTIV - toate procesele
     * din MPI_COMM_WORLD trebuie sa apeleze aceasta functie!
     * Procesele care nu apartin grupului dat primesc MPI_COMM_NULL.
     *
     * Un comunicator = grup + context unic. Contextul diferit permite
     * ca mesajele pe new_comm sa nu interfere cu cele pe COMM_WORLD.      */
    MPI_Comm my_comm;
    MPI_Comm_create(MPI_COMM_WORLD, my_group, &my_comm);

    /* Obtinem rangul procesului in contextul noului grup.
     * Exemplu: procesul cu rank=5 in COMM_WORLD are new_rank=1
     * in grupul {4,5}.                                                    */
    int new_rank;
    MPI_Group_rank(my_group, &new_rank);

    /* ==========================================================
     * PASUL 4: Fiecare grup efectueaza operatia sa SIMULTAN
     *
     * Cele 4 MPI_Allreduce ruleaza independent deoarece
     * fiecare foloseste un comunicator diferit (my_comm).
     * MPI poate suprapune executia lor in timp.
     *
     * MPI_Allreduce combina valorile de la toate procesele
     * din comunicator si distribuie rezultatul tuturor.
     * ========================================================== */
    long long result_sum  = 0;
    long long result_prod = 1;
    int       result_min  = 0;
    int       result_max  = 0;

    switch (group_id) {

        case 0:
            /* Grup 0: SUMA - aduna local_sum de la procesele 0 si 1
             * Procesul 0 are local_sum=3 (1+2), procesul 1 are 7 (3+4)
             * Rezultat: 3+7 = 10 = suma elementelor {1,2,3,4}            */
            MPI_Allreduce(&local_sum, &result_sum, 1,
                          MPI_LONG_LONG, MPI_SUM, my_comm);
            if (new_rank == 0)
                printf("[Grup SUMA  ] Suma   elemente {1..4}   = %lld\n",
                       result_sum);
            break;

        case 1:
            /* Grup 1: PRODUS - inmulteste local_prod de la proc 2 si 3
             * Proc 2 are local_prod=30 (5*6), proc 3 are 56 (7*8)
             * Rezultat: 30*56 = 1680 = produsul {5,6,7,8}               */
            MPI_Allreduce(&local_prod, &result_prod, 1,
                          MPI_LONG_LONG, MPI_PROD, my_comm);
            if (new_rank == 0)
                printf("[Grup PRODUS] Produs elemente {5..8}   = %lld\n",
                       result_prod);
            break;

        case 2:
            /* Grup 2: MINIM - gaseste minimul intre proc 4 si 5
             * Proc 4 are local_min=9,  proc 5 are local_min=11
             * Rezultat: min(9, 11) = 9                                   */
            MPI_Allreduce(&local_min, &result_min, 1,
                          MPI_INT, MPI_MIN, my_comm);
            if (new_rank == 0)
                printf("[Grup MINIM ] Minim  elemente {9..12}  = %d\n",
                       result_min);
            break;

        case 3:
            /* Grup 3: MAXIM - gaseste maximul intre proc 6 si 7
             * Proc 6 are local_max=14, proc 7 are local_max=16
             * Rezultat: max(14, 16) = 16                                 */
            MPI_Allreduce(&local_max, &result_max, 1,
                          MPI_INT, MPI_MAX, my_comm);
            if (new_rank == 0)
                printf("[Grup MAXIM ] Maxim  elemente {13..16} = %d\n",
                       result_max);
            break;
    }

    /* ==========================================================
     * PASUL 5: Colectarea rezultatelor finale la procesul 0
     *
     * Liderul fiecarui grup (new_rank == 0) trimite rezultatul
     * partial la procesul 0 din COMM_WORLD, care va combina
     * rezultatele pentru a obtine valorile globale ale vectorului.
     *
     * Combinare:
     *   global_sum  = sum_G0 + sum_G1 + sum_G2 + sum_G3
     *   global_prod = prod_G0 * prod_G1 * prod_G2 * prod_G3
     *   global_min  = min(min_G0, min_G1, min_G2, min_G3)
     *   global_max  = max(max_G0, max_G1, max_G2, max_G3)
     * ========================================================== */

    /* Fiecare grup calculeaza si partial_sum/prod/min/max local
     * pentru a permite combinarea globala. Fiecare proces calculeaza
     * toate cele 4 valori, indiferent de grupul sau.                     */
    if (rank == 0) {
        /* Procesul 0 este liderul Grupului 0 (SUMA).
         * Primeste rezultatele de la liderii celorlalte grupuri.         */
        long long partial_sums[4];
        long long partial_prods[4];
        int       partial_mins[4];
        int       partial_maxs[4];

        /* Valorile proprii (din Grupul 0) */
        partial_sums[0]  = result_sum;
        partial_prods[0] = local_prod;       /* produsul local al proc 0 */
        partial_mins[0]  = local_min;
        partial_maxs[0]  = local_max;

        /* Primeste de la liderul Grupului 1 (rank global = 2): produs  */
        MPI_Recv(&partial_sums[1],  1, MPI_LONG_LONG, 2, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&partial_prods[1], 1, MPI_LONG_LONG, 2, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&partial_mins[1],  1, MPI_INT,       2, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&partial_maxs[1],  1, MPI_INT,       2, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Primeste de la liderul Grupului 2 (rank global = 4): minim   */
        MPI_Recv(&partial_sums[2],  1, MPI_LONG_LONG, 4, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&partial_prods[2], 1, MPI_LONG_LONG, 4, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&partial_mins[2],  1, MPI_INT,       4, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&partial_maxs[2],  1, MPI_INT,       4, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Primeste de la liderul Grupului 3 (rank global = 6): maxim   */
        MPI_Recv(&partial_sums[3],  1, MPI_LONG_LONG, 6, 10, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&partial_prods[3], 1, MPI_LONG_LONG, 6, 11, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&partial_mins[3],  1, MPI_INT,       6, 12, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(&partial_maxs[3],  1, MPI_INT,       6, 13, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        /* Combina rezultatele partiale */
        long long global_sum  = 0;
        long long global_prod = 1;
        int       global_min  = INT_MAX;
        int       global_max  = INT_MIN;

        for (int g = 0; g < 4; g++) {
            global_sum  += partial_sums[g];
            global_prod *= partial_prods[g];
            if (partial_mins[g] < global_min) global_min = partial_mins[g];
            if (partial_maxs[g] > global_max) global_max = partial_maxs[g];
        }

        printf("\n=== REZULTATE GLOBALE pentru vectorul [1..%d] ===\n", N);
        printf("Suma    = %lld  (asteptat: %d)\n", global_sum,  N*(N+1)/2);
        printf("Produs  = %lld\n", global_prod);
        printf("Minim   = %d   (asteptat: 1)\n",  global_min);
        printf("Maxim   = %d  (asteptat: %d)\n",  global_max, N);

    } else if (new_rank == 0) {
        /* Liderii grupurilor 1, 2, 3 (rank global 2, 4, 6)
         * trimit TOATE valorile partiale la procesul 0,
         * pentru ca acesta sa poata combina toate operatiile.           */
        long long my_partial_sum  = local_sum;
        long long my_partial_prod = local_prod;
        int       my_partial_min  = local_min;
        int       my_partial_max  = local_max;

        /* Trimite catre procesul 0 toate valorile partiale ale grupului */
        long long group_partial_sum  = 0;
        long long group_partial_prod = 1;
        int       group_partial_min  = INT_MAX;
        int       group_partial_max  = INT_MIN;

        /* Reduce in cadrul grupului pentru a obtine valorile grupului   */
        MPI_Reduce(&my_partial_sum,  &group_partial_sum,  1, MPI_LONG_LONG, MPI_SUM,  0, my_comm);
        MPI_Reduce(&my_partial_prod, &group_partial_prod, 1, MPI_LONG_LONG, MPI_PROD, 0, my_comm);
        MPI_Reduce(&my_partial_min,  &group_partial_min,  1, MPI_INT,       MPI_MIN,  0, my_comm);
        MPI_Reduce(&my_partial_max,  &group_partial_max,  1, MPI_INT,       MPI_MAX,  0, my_comm);

        /* Liderul grupului (new_rank==0) trimite la proc 0 */
        MPI_Send(&group_partial_sum,  1, MPI_LONG_LONG, 0, 10, MPI_COMM_WORLD);
        MPI_Send(&group_partial_prod, 1, MPI_LONG_LONG, 0, 11, MPI_COMM_WORLD);
        MPI_Send(&group_partial_min,  1, MPI_INT,       0, 12, MPI_COMM_WORLD);
        MPI_Send(&group_partial_max,  1, MPI_INT,       0, 13, MPI_COMM_WORLD);

    } else {
        /* Procesele non-lideri (new_rank != 0) participa la Reduce
         * din cadrul grupului, dar nu trimit direct la procesul 0.
         * MPI_Reduce este colectiv - TOTI din grup trebuie sa apeleze! */
        long long dummy_ll;
        int       dummy_i;
        MPI_Reduce(&local_sum,  &dummy_ll, 1, MPI_LONG_LONG, MPI_SUM,  0, my_comm);
        MPI_Reduce(&local_prod, &dummy_ll, 1, MPI_LONG_LONG, MPI_PROD, 0, my_comm);
        MPI_Reduce(&local_min,  &dummy_i,  1, MPI_INT,       MPI_MIN,  0, my_comm);
        MPI_Reduce(&local_max,  &dummy_i,  1, MPI_INT,       MPI_MAX,  0, my_comm);
    }

    /* ==========================================================
     * PASUL 6: Eliberarea resurselor MPI
     *
     * Ordinea corecta: comunicator INAINTE de grup.
     * Un comunicator mentine o referinta interna la grupul sau,
     * deci eliberarea grupului inainte ar corupe starea interna.
     * ========================================================== */
    free(local_data);
    MPI_Comm_free(&my_comm);      /* eliberam comunicatorul grupului      */
    MPI_Group_free(&my_group);    /* eliberam subgrupul creat             */
    MPI_Group_free(&world_group); /* eliberam referinta la grupul global  */

    MPI_Finalize();
    return 0;
}
