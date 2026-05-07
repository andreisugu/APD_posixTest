#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 8
#define NPROCS 4
#define INF 99999

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != NPROCS)
    {
        if (rank == 0)
            printf("Eroare: rulati cu exact %d procese!\n", NPROCS);
        MPI_Finalize();
        return 1;
    }

    int rows_per_proc = N / NPROCS;
    int *D = NULL;
    int *local_D = (int *)malloc(rows_per_proc * N * sizeof(int));
    int *row_k = (int *)malloc(N * sizeof(int));

    if (rank == 0)
    {
        D = (int *)malloc(N * N * sizeof(int));

        printf("Matricea initiala a distantelor (INF=%d):\n", INF);
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                if (i == j)
                    D[i * N + j] = 0;
                else if (abs(i - j) == 1)
                    D[i * N + j] = 1;
                else
                    D[i * N + j] = INF;

                if (D[i * N + j] == INF)
                    printf("  INF");
                else
                    printf(" %4d", D[i * N + j]);
            }
            printf("\n");
        }
        printf("\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double start_time = MPI_Wtime();
    MPI_Scatter(D, rows_per_proc * N, MPI_INT,
                local_D, rows_per_proc * N, MPI_INT,
                0, MPI_COMM_WORLD);

    for (int k = 0; k < N; k++)
    {
        int owner_rank = k / rows_per_proc;
        if (rank == owner_rank)
        {
            int local_k_index = k % rows_per_proc;
            for (int j = 0; j < N; j++)
            {
                row_k[j] = local_D[local_k_index * N + j];
            }
        }

        MPI_Bcast(row_k, N, MPI_INT, owner_rank, MPI_COMM_WORLD);
        for (int i = 0; i < rows_per_proc; i++)
        {
            for (int j = 0; j < N; j++)
            {
                int dist_ik = local_D[i * N + k];
                int dist_kj = row_k[j];

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

    MPI_Gather(local_D, rows_per_proc * N, MPI_INT,
               D, rows_per_proc * N, MPI_INT,
               0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double end_time = MPI_Wtime();

    if (rank == 0)
    {
        printf("Matricea finala a distantelor minime:\n");
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                printf(" %4d", D[i * N + j]);
            }
            printf("\n");
        }

        printf("\n=== TIMP DE EXECUTIE ===\n");
        printf("Timp MPI Roy-Floyd: %f secunde\n", end_time - start_time);
        
        free(D);
    }

    free(local_D);
    free(row_k);

    MPI_Finalize();
    return 0;
}