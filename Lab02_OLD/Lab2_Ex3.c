/*
 * Exercitiul 3 - Laborator 2 MPI
 * Inversarea culorilor unei imagini reprezentate ca tablou 1D de valori RGBA.
 * - Fiecare componenta R, G, B este inversata: new_value = 255 - old_value.
 * - Componenta Alpha ramane neschimbata.
 * - Distributia pixelilor se face cu MPI_Scatter; colectarea cu MPI_Gather.
 *
 * Semnatura solve() primeste rank si size din main (mai clara, fara apeluri MPI interne).
 * Toate procesele aloca bufferul image (necesar pentru MPI_Scatter).
 * Rezultatul final este stocat in matricea image (pe procesul 0).
 *
 * Exemplu 1: image=[255,0,128,255, 0,255,0,255], w=1, h=2
 *            => [0,255,127,255, 255,0,255,255]
 * Exemplu 2: image=[10,20,30,255, 100,150,200,255], w=2, h=1
 *            => [245,235,225,255, 155,105,55,255]
 *
 * Compilare: mpicc Lab2_Ex3.c -o Lab2_Ex3
 * Rulare:    mpirun -np 4 ./Lab2_Ex3
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef unsigned char uchar;

/* ------------------------------------------------------------------ */
/* Inverseaza culorile imaginii in-place (paralel cu MPI).            */
/* rank si size sunt transmisi din main pentru claritate.             */
/* ------------------------------------------------------------------ */
void solve(uchar *image, int width, int height, int rank, int size) {
    int total_pixels    = width * height;
    int pixels_per_proc = total_pixels / size;
    int bytes_per_proc  = pixels_per_proc * 4;

    /*
     * Daca avem mai putini pixeli decat procese, rank 0 proceseaza
     * totul serial (MPI_Scatter cu 0 bytes nu face nimic util).
     */
    if (pixels_per_proc == 0) {
        if (rank == 0) {
            for (int i = 0; i < total_pixels; i++) {
                image[i * 4 + 0] = (uchar)(255 - image[i * 4 + 0]);
                image[i * 4 + 1] = (uchar)(255 - image[i * 4 + 1]);
                image[i * 4 + 2] = (uchar)(255 - image[i * 4 + 2]);
            }
        }
        return;
    }

    uchar *local_buf = (uchar *)malloc(bytes_per_proc * sizeof(uchar));
    if (!local_buf) {
        fprintf(stderr, "Rank %d: malloc esuat\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    /* Scatter: distributia blocurilor de pixeli */
    MPI_Scatter(image, bytes_per_proc, MPI_UNSIGNED_CHAR,
                local_buf, bytes_per_proc, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    /* Inversare locala: R,G,B = 255 - val; A ramane neschimbat */
    for (int i = 0; i < bytes_per_proc; i += 4) {
        local_buf[i + 0] = (uchar)(255 - local_buf[i + 0]); /* R */
        local_buf[i + 1] = (uchar)(255 - local_buf[i + 1]); /* G */
        local_buf[i + 2] = (uchar)(255 - local_buf[i + 2]); /* B */
        /* local_buf[i + 3] neschimbat (Alpha)                    */
    }

    /* Gather: colectare rezultate in image[] pe procesul 0 */
    MPI_Gather(local_buf, bytes_per_proc, MPI_UNSIGNED_CHAR,
               image,     bytes_per_proc, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);

    /* Rank 0 proceseaza pixelii ramasi (total_pixels % size) */
    if (rank == 0) {
        int remainder = total_pixels % size;
        int offset    = pixels_per_proc * size;
        for (int i = offset; i < offset + remainder; i++) {
            image[i * 4 + 0] = (uchar)(255 - image[i * 4 + 0]);
            image[i * 4 + 1] = (uchar)(255 - image[i * 4 + 1]);
            image[i * 4 + 2] = (uchar)(255 - image[i * 4 + 2]);
        }
    }

    free(local_buf);
}

/* ------------------------------------------------------------------ */
int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ---- Exemplul 1 ---- */
    if (rank == 0)
        printf("=== Exercitiul 3: Inversare culori imagine RGBA ===\n\n"
               "--- Exemplul 1 ---\n");
    {
        int width = 1, height = 2;
        int total_bytes = width * height * 4;

        /* Toate procesele aloca bufferul – necesar pentru MPI_Scatter */
        uchar *image = (uchar *)malloc(total_bytes * sizeof(uchar));

        if (rank == 0) {
            uchar init[] = {255, 0, 128, 255,   0, 255, 0, 255};
            memcpy(image, init, total_bytes);
            printf("Input:    ");
            for (int i = 0; i < total_bytes; i++) printf("%d ", image[i]);
            printf("\n");
        }

        solve(image, width, height, rank, size);

        if (rank == 0) {
            printf("Output:   ");
            for (int i = 0; i < total_bytes; i++) printf("%d ", image[i]);
            printf("\nAsteptat: 0 255 127 255 255 0 255 255\n");
        }
        free(image);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* ---- Exemplul 2 ---- */
    if (rank == 0) printf("\n--- Exemplul 2 ---\n");
    {
        int width = 2, height = 1;
        int total_bytes = width * height * 4;

        uchar *image = (uchar *)malloc(total_bytes * sizeof(uchar));

        if (rank == 0) {
            uchar init[] = {10, 20, 30, 255,   100, 150, 200, 255};
            memcpy(image, init, total_bytes);
            printf("Input:    ");
            for (int i = 0; i < total_bytes; i++) printf("%d ", image[i]);
            printf("\n");
        }

        solve(image, width, height, rank, size);

        if (rank == 0) {
            printf("Output:   ");
            for (int i = 0; i < total_bytes; i++) printf("%d ", image[i]);
            printf("\nAsteptat: 245 235 225 255 155 105 55 255\n");
        }
        free(image);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* ---- Test imagine 4x4 generata automat ---- */
    if (rank == 0) printf("\n--- Test imagine 4x4 (generata automat) ---\n");
    {
        int width = 4, height = 4;
        int total_pixels = width * height;   /* 16 pixeli */
        int total_bytes  = total_pixels * 4; /* 64 octeti  */

        uchar *image = (uchar *)malloc(total_bytes * sizeof(uchar));

        if (rank == 0) {
            for (int i = 0; i < total_pixels; i++) {
                image[i * 4 + 0] = (uchar)(i * 4);         /* R */
                image[i * 4 + 1] = (uchar)(i * 4 + 1);     /* G */
                image[i * 4 + 2] = (uchar)(i * 4 + 2);     /* B */
                image[i * 4 + 3] = 255;                     /* A */
            }
            printf("Input R-values:  ");
            for (int i = 0; i < total_pixels; i++) printf("%3d ", image[i * 4]);
            printf("\n");
        }

        solve(image, width, height, rank, size);

        if (rank == 0) {
            printf("Output R-values: ");
            for (int i = 0; i < total_pixels; i++) printf("%3d ", image[i * 4]);
            printf("\n");
            int ok = 1;
            for (int i = 0; i < total_pixels; i++)
                if (image[i * 4 + 3] != 255) { ok = 0; break; }
            printf("Alpha intact: %s\n", ok ? "DA" : "NU");
        }
        free(image);
    }

    MPI_Finalize();
    return 0;
}
