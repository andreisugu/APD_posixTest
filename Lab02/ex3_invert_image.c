/*
 * Exercitiul 3: Inversarea culorilor unei imagini RGBA
 * - Imaginea este un tablou 1D: width * height * 4 octeti (RGBA)
 * - Inversare: R = 255-R, G = 255-G, B = 255-B, A ramine neschimbat
 * - MPI_Scatter  : distribuie blocuri de pixeli catre procese
 * - MPI_Gather   : aduna pixelii inversati inapoi in procesul 0
 * - Rezultatul final este stocat in acelasi array (image[])
 *
 * Compilare: mpicc -o ex3 ex3_invert_image.c
 * Rulare:    mpirun -np 4 ./ex3
 *
 * NOTA: (width * height) trebuie sa fie multiplu de numarul de procese
 *       pentru a se potrivi cu MPI_Scatter/Gather simplu.
 *       In productie s-ar folosi MPI_Scatterv.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "mpi.h"

typedef unsigned char uchar;

/* ------------------------------------------------------------------ */
/* Functia solve – semnatura RAMANE NESCHIMBATA conform cerintei       */
/* image   : tabloul RGBA de intrare/iesire                            */
/* width   : latimea imaginii in pixeli                                */
/* height  : inaltimea imaginii in pixeli                              */
/* rank, size: informatii MPI transmise din main                       */
/* ------------------------------------------------------------------ */
void solve(uchar *image, int width, int height, int rank, int size) {
    int total_pixels  = width * height;
    int total_bytes   = total_pixels * 4;   /* 4 canale RGBA */

    /*
     * Impartim totalul de PIXELI intre procese.
     * Fiecare pixel = 4 octeti  =>  fiecare proces lucreaza pe
     * pixels_per_proc * 4 octeti.
     *
     * CERINTA: total_pixels % size == 0 (ajustat mai jos daca e nevoie)
     */
    int pixels_per_proc = total_pixels / size;
    int bytes_per_proc  = pixels_per_proc * 4;

    uchar *local_buf = (uchar *)malloc(bytes_per_proc * sizeof(uchar));

    /* --- Scatter: distribuie bytes catre toti procesii --- */
    MPI_Scatter(image, bytes_per_proc, MPI_UNSIGNED_CHAR,
                local_buf, bytes_per_proc, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    /* --- Inversare locala: R,G,B = 255 - val; A ramine neschimbat --- */
    for (int i = 0; i < bytes_per_proc; i += 4) {
        local_buf[i + 0] = 255 - local_buf[i + 0]; /* R */
        local_buf[i + 1] = 255 - local_buf[i + 1]; /* G */
        local_buf[i + 2] = 255 - local_buf[i + 2]; /* B */
        /* local_buf[i + 3] = local_buf[i + 3]; */  /* A neschimbat */
    }

    /* --- Gather: aduna rezultatele in image[] pe procesul 0 --- */
    MPI_Gather(local_buf, bytes_per_proc, MPI_UNSIGNED_CHAR,
               image, bytes_per_proc, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);

    free(local_buf);
}

/* ------------------------------------------------------------------ */
int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    /* ---- Exemplul 1 ---- */
    if (rank == 0) {
        printf("=== Exercitiul 3: Inversare culori imagine RGBA ===\n\n");
        printf("--- Exemplul 1 ---\n");
    }
    {
        int width = 1, height = 2;
        int total_bytes = width * height * 4;

        uchar *image = NULL;
        if (rank == 0) {
            image = (uchar *)malloc(total_bytes * sizeof(uchar));
            uchar init[] = {255, 0, 128, 255,   0, 255, 0, 255};
            memcpy(image, init, total_bytes);

            printf("Input:  ");
            for (int i = 0; i < total_bytes; i++) printf("%d ", image[i]);
            printf("\n");
        } else {
            /* Procesele non-zero au nevoie de buffer pentru Scatter */
            image = (uchar *)malloc(total_bytes * sizeof(uchar));
        }

        solve(image, width, height, rank, size);

        if (rank == 0) {
            printf("Output: ");
            for (int i = 0; i < total_bytes; i++) printf("%d ", image[i]);
            printf("\n");
            printf("Asteptat: 0 255 127 255 255 0 255 255\n\n");
        }
        free(image);
    }

    /* Bariera pentru a separa vizual cele doua exemple */
    MPI_Barrier(MPI_COMM_WORLD);

    /* ---- Exemplul 2 ---- */
    if (rank == 0) {
        printf("--- Exemplul 2 ---\n");
    }
    {
        int width = 2, height = 1;
        int total_bytes = width * height * 4;

        uchar *image = NULL;
        if (rank == 0) {
            image = (uchar *)malloc(total_bytes * sizeof(uchar));
            uchar init[] = {10, 20, 30, 255,   100, 150, 200, 255};
            memcpy(image, init, total_bytes);

            printf("Input:  ");
            for (int i = 0; i < total_bytes; i++) printf("%d ", image[i]);
            printf("\n");
        } else {
            image = (uchar *)malloc(total_bytes * sizeof(uchar));
        }

        solve(image, width, height, rank, size);

        if (rank == 0) {
            printf("Output: ");
            for (int i = 0; i < total_bytes; i++) printf("%d ", image[i]);
            printf("\n");
            printf("Asteptat: 245 235 225 255 155 105 55 255\n\n");
        }
        free(image);
    }

    /* ---- Test cu imagine mai mare (4x4) ---- */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("--- Test imagine 4x4 (generata automat) ---\n");
    }
    {
        int width = 4, height = 4;
        int total_pixels = width * height;   /* 16 pixeli */
        int total_bytes  = total_pixels * 4; /* 64 octeti  */

        uchar *image = NULL;
        if (rank == 0) {
            image = (uchar *)malloc(total_bytes * sizeof(uchar));
            for (int i = 0; i < total_pixels; i++) {
                image[i * 4 + 0] = (uchar)(i * 4);      /* R */
                image[i * 4 + 1] = (uchar)(i * 4 + 1);  /* G */
                image[i * 4 + 2] = (uchar)(i * 4 + 2);  /* B */
                image[i * 4 + 3] = 255;                  /* A */
            }
        } else {
            image = (uchar *)malloc(total_bytes * sizeof(uchar));
        }

        solve(image, width, height, rank, size);

        if (rank == 0) {
            printf("Verificare (primii 3 pixeli):\n");
            for (int i = 0; i < 3; i++) {
                uchar r = image[i*4], g = image[i*4+1],
                      b = image[i*4+2], a = image[i*4+3];
                printf("  Pixel %d: R=%d G=%d B=%d A=%d\n", i, r, g, b, a);
            }
            printf("Toate componentele A ar trebui sa fie 255 (neschimbate).\n");
            free(image);
        } else {
            free(image);
        }
    }

    MPI_Finalize();
    return 0;
}
