/*
 * ============================================================
 * EXERCITIUL 3 - MPI Pipeline cu 3 Grupuri de Procese
 * ============================================================
 * Problema: Se da un vector de N numere reale. Se creeaza 3
 * grupuri de procese care formeaza un pipeline:
 *
 *   Grupul 1 (proc 0, 1): NORMALIZEAZA valorile in [0, 1]
 *                          formula: x_norm = (x - min) / (max - min)
 *
 *   Grupul 2 (proc 2, 3): APLICA SQRT fiecarui element
 *
 *   Grupul 3 (proc 4, 5): SORTEAZA rezultatul final
 *
 * Fluxul datelor (pipeline):
 *   [Date brute] -> Grup1 -> [Normalizat] -> Grup2 -> [sqrt] -> Grup3 -> [Sortat]
 *
 * Comunicare inter-grup: prin MPI_COMM_WORLD cu Send/Recv
 * intre procesele corespunzatoare din grupuri consecutive.
 *
 * Compilare: mpicc -o ex3 ex3_pipeline_grupuri.c -lm
 * Rulare:    mpirun -np 6 ./ex3
 * ============================================================
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define N      6    /* marimea vectorului (multiplu de procs_per_group) */
#define NPROCS 6    /* 2 procese/grup × 3 grupuri                       */

/* Tag-uri MPI pentru a distinge mesajele intre grupuri                  */
#define TAG_G1_TO_G2  100  /* mesaj de la Grupul 1 la Grupul 2          */
#define TAG_G2_TO_G3  200  /* mesaj de la Grupul 2 la Grupul 3          */

/* Sortare simpla (bubble sort) */
void bubble_sort(double *arr, int n) {
    for (int i = 0; i < n - 1; i++)
        for (int j = 0; j < n - i - 1; j++)
            if (arr[j] > arr[j + 1]) {
                double tmp = arr[j];
                arr[j]     = arr[j + 1];
                arr[j + 1] = tmp;
            }
}

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

    /* ==========================================================
     * PASUL 1: Definim cele 3 grupuri de procese
     *
     * Impartire uniforma: group_id = rank / procs_per_group
     *   Grup 0 (G1): proc {0, 1} -> normalizare
     *   Grup 1 (G2): proc {2, 3} -> sqrt
     *   Grup 2 (G3): proc {4, 5} -> sortare
     *
     * Fiecare grup are propriul comunicator (my_comm) folosit
     * pentru operatii colective INTRA-grup (reduceri, gather).
     * Comunicarea INTER-grup se face prin MPI_COMM_WORLD.
     * ========================================================== */
    int procs_per_group = 2;
    int group_id = rank / procs_per_group;  /* 0, 1, sau 2               */

    /* Array cu rangurile proceselor din fiecare grup                    */
    int ranks_g[3][2] = {
        {0, 1},   /* Grupul 1 (G1): normalizare  */
        {2, 3},   /* Grupul 2 (G2): sqrt         */
        {4, 5}    /* Grupul 3 (G3): sortare      */
    };

    /* Extragem grupul global si cream subgrupul local */
    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    /* MPI_Group_incl: selecteaza din world_group procesele cu
     * rangurile din ranks_g[group_id] si formeaza un nou grup.          */
    MPI_Group my_group;
    MPI_Group_incl(world_group, procs_per_group,
                   ranks_g[group_id], &my_group);

    /* MPI_Comm_create: creeaza comunicatorul pentru grupul local.
     * APEL COLECTIV - toate procesele din COMM_WORLD il apeleaza!       */
    MPI_Comm my_comm;
    MPI_Comm_create(MPI_COMM_WORLD, my_group, &my_comm);

    /* Rangul in cadrul grupului propriu (0 sau 1)                       */
    int new_rank;
    MPI_Group_rank(my_group, &new_rank);

    /* ==========================================================
     * Distributia elementelor:
     * Fiecare proces detine elems_per_proc = N / NPROCS = 1 element.
     * Intr-un grup de 2 procese, grupul detine 2 elemente total.
     *
     * Vectorul initial (doar la G1):
     *   proc 0: data[0] = 10.0   proc 1: data[0] = 2.0
     *   (G2 si G3 primesc date prin comunicare pipeline)
     * ========================================================== */
    int elems_per_proc = N / NPROCS;   /* = 1 element per proces         */
    double *data = malloc(elems_per_proc * sizeof(double));

    /* Date initiale brute pentru intreg vectorul: */
    double initial_data[N] = {10.0, 2.0, 8.0, 4.0, 6.0, 5.0};
    /* Valorile sunt: {10, 2, 8, 4, 6, 5} -> dupa normalizare, sqrt, sort */

    /* ==========================================================
     * PASUL 2: GRUPUL 1 - Normalizare in intervalul [0, 1]
     *
     * Formula: x_norm = (x - global_min) / (global_max - global_min)
     *
     * Algoritmul:
     *   a) Fiecare proc din G1 are un element din vector
     *   b) Calculam min si max global prin MPI_Allreduce IN G1
     *   c) Fiecare proc normalizeaza elementul sau
     *   d) Proc 0 si 1 din G1 trimit datele normalizate catre
     *      proc 2 si 3 din G2 (corespunzator, prin COMM_WORLD)
     * ========================================================== */
    if (group_id == 0) {

        /* Initializare: proc cu new_rank r are elementul initial_data[r] */
        for (int i = 0; i < elems_per_proc; i++) {
            data[i] = initial_data[new_rank * elems_per_proc + i];
        }

        printf("[G1, proc %d] Valoare initiala: %.2f\n", rank, data[0]);

        /* Calculam minimul si maximul global in cadrul G1 */
        double local_min = data[0];
        double local_max = data[0];
        for (int i = 1; i < elems_per_proc; i++) {
            if (data[i] < local_min) local_min = data[i];
            if (data[i] > local_max) local_max = data[i];
        }

        double global_min, global_max;
        /* MPI_Allreduce pe MY_COMM (nu COMM_WORLD!) - operatia se
         * desfasoara DOAR intre procesele din G1.
         * Rezultatul (min/max global din G1) este distribuit
         * tuturor membrilor grupului.                                   */
        MPI_Allreduce(&local_min, &global_min, 1,
                      MPI_DOUBLE, MPI_MIN, my_comm);
        MPI_Allreduce(&local_max, &global_max, 1,
                      MPI_DOUBLE, MPI_MAX, my_comm);

        /* Normalizam elementele: x_norm = (x - min) / (max - min)      */
        for (int i = 0; i < elems_per_proc; i++) {
            data[i] = (data[i] - global_min) / (global_max - global_min);
        }

        printf("[G1, proc %d] Dupa normalizare (min=%.1f, max=%.1f): %.4f\n",
               rank, global_min, global_max, data[0]);

        /* Trimite datele normalizate catre Grupul 2 (G2).
         * Corespondenta: new_rank 0 (proc 0) -> proc 2 din G2
         *                new_rank 1 (proc 1) -> proc 3 din G2
         * Destul: rang global in G2 = 2 + new_rank                     */
        int dest_g2 = 2 + new_rank;
        MPI_Send(data, elems_per_proc, MPI_DOUBLE,
                 dest_g2, TAG_G1_TO_G2, MPI_COMM_WORLD);

        printf("[G1, proc %d] -> Trimis %.4f catre proc %d (G2)\n",
               rank, data[0], dest_g2);
    }

    /* ==========================================================
     * PASUL 3: GRUPUL 2 - Aplicare functie sqrt
     *
     * Algoritmul:
     *   a) Primeste elementele normalizate de la G1
     *      (proc 2 de la proc 0, proc 3 de la proc 1)
     *   b) Aplica sqrt fiecarui element
     *   c) Trimite rezultatele catre G3
     * ========================================================== */
    else if (group_id == 1) {

        /* Sursa: rangul global corespunzator din G1 = new_rank          */
        int src_g1 = new_rank;
        MPI_Recv(data, elems_per_proc, MPI_DOUBLE,
                 src_g1, TAG_G1_TO_G2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("[G2, proc %d] Primit de la proc %d: %.4f\n",
               rank, src_g1, data[0]);

        /* Aplicam sqrt fiecarui element                                 */
        for (int i = 0; i < elems_per_proc; i++) {
            data[i] = sqrt(data[i]);
        }

        printf("[G2, proc %d] Dupa sqrt: %.4f\n", rank, data[0]);

        /* Trimite catre G3.
         * Corespondenta: new_rank 0 (proc 2) -> proc 4 din G3
         *                new_rank 1 (proc 3) -> proc 5 din G3
         * Dest: rang global in G3 = 4 + new_rank                       */
        int dest_g3 = 4 + new_rank;
        MPI_Send(data, elems_per_proc, MPI_DOUBLE,
                 dest_g3, TAG_G2_TO_G3, MPI_COMM_WORLD);

        printf("[G2, proc %d] -> Trimis %.4f catre proc %d (G3)\n",
               rank, data[0], dest_g3);
    }

    /* ==========================================================
     * PASUL 4: GRUPUL 3 - Sortare finala
     *
     * Algoritmul:
     *   a) Primeste elementele de la G2
     *   b) Aduna toate elementele la procesul 0 din G3 (MPI_Gather)
     *   c) Proc 0 din G3 sorteaza si afiseaza vectorul final
     * ========================================================== */
    else { /* group_id == 2 */

        /* Sursa: rangul global corespunzator din G2 = 2 + new_rank      */
        int src_g2 = 2 + new_rank;
        MPI_Recv(data, elems_per_proc, MPI_DOUBLE,
                 src_g2, TAG_G2_TO_G3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("[G3, proc %d] Primit de la proc %d: %.4f\n",
               rank, src_g2, data[0]);

        /* Adunam toate elementele grupului la liderul G3 (new_rank = 0)
         * MPI_Gather colecteaza elems_per_proc elemente de la fiecare
         * proces din my_comm si le concateneaza la radacina (new_rank 0).
         * Buffer-ul all_data are N/3 * procs_per_group = 2 elemente.   */
        int group_n = elems_per_proc * procs_per_group;  /* 2 elemente   */
        double *all_data = NULL;

        if (new_rank == 0) {
            all_data = malloc(group_n * sizeof(double));
        }

        /* MPI_Gather INTRA-grup: fiecare proc din G3 trimite 1 element
         * la new_rank 0 al grupului G3.                                 */
        MPI_Gather(data,           /* buffer sursa (1 element)            */
                   elems_per_proc, /* nr. elemente de trimis              */
                   MPI_DOUBLE,
                   all_data,       /* buffer destinatie (la new_rank=0)   */
                   elems_per_proc, /* nr. elemente asteptate de la fiecare*/
                   MPI_DOUBLE,
                   0,              /* radacina grupului = new_rank 0      */
                   my_comm);       /* comunicatorul GRUPULUI G3 !!!       */

        if (new_rank == 0) {
            /* Sortam vectorul complet al grupului */
            bubble_sort(all_data, group_n);

            printf("\n[G3, proc %d] Vectorul final dupa sortare:\n", rank);
            for (int i = 0; i < group_n; i++) {
                printf("  [%d] = %.4f\n", i, all_data[i]);
            }
            free(all_data);
        }
    }

    /* ==========================================================
     * PASUL 5: Bariera de sincronizare (optional, pentru output curat)
     * MPI_Barrier asigura ca toate procesele au terminat
     * inainte ca vreun proces sa continue.
     * ========================================================== */
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n=== PIPELINE COMPLET ===\n");
        printf("Date brute   : {10, 2, 8, 4, 6, 5}\n");
        printf("Dupa G1 norm : valorile aduse in [0, 1]\n");
        printf("Dupa G2 sqrt : sqrt aplicat fiecarui element\n");
        printf("Dupa G3 sort : vectorul sortat crescator\n");
    }

    /* ==========================================================
     * PASUL 6: Eliberarea resurselor
     * ========================================================== */
    free(data);
    MPI_Comm_free(&my_comm);
    MPI_Group_free(&my_group);
    MPI_Group_free(&world_group);

    MPI_Finalize();
    return 0;
}
