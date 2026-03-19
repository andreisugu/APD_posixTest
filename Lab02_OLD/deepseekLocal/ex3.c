#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    unsigned char *image = NULL;
    int width, height;
    int total_pixels;

    // Read input on root process
    if (rank == 0) {
        printf("Enter image width: ");
        scanf("%d", &width);
        printf("Enter image height: ");
        scanf("%d", &height);
        total_pixels = width * height;
        image = (unsigned char *)malloc(total_pixels * 4 * sizeof(unsigned char));
        printf("Enter image data ("
               "%d pixels, 4 components each, total %d elements): \n",
               total_pixels, total_pixels * 4);
        fread(image, sizeof(unsigned char), total_pixels * 4, stdin);
    }

    // Broadcast width and height to all processes
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    total_pixels = width * height;
    int pixels_per_process = (total_pixels + size - 1) / size;
    int elements_per_process = pixels_per_process * 4;

    // Allocate memory for local array
    unsigned char *local_image = (unsigned char *)malloc(elements_per_process * sizeof(unsigned char));

    // Scatter the image data to all processes
    MPI_Scatter(image, elements_per_process, MPI_UNSIGNED_CHAR,
                local_image, elements_per_process, MPI_UNSIGNED_CHAR,
                0, MPI_COMM_WORLD);

    // Invert the colors in the local array
    for (int i = 0; i < pixels_per_process; ++i) {
        int index = i * 4;
        local_image[index] = 255 - local_image[index]; // Invert R
        local_image[index + 1] = 255 - local_image[index + 1]; // Invert G
        local_image[index + 2] = 255 - local_image[index + 2]; // Invert B
        // Alpha remains unchanged
    }

    // Gather the inverted data back to the root process
    unsigned char *result = NULL;
    if (rank == 0) {
        result = (unsigned char *)malloc(total_pixels * 4 * sizeof(unsigned char));
    }
    MPI_Gather(local_image, elements_per_process, MPI_UNSIGNED_CHAR,
               result, elements_per_process, MPI_UNSIGNED_CHAR,
               0, MPI_COMM_WORLD);

    // Output the result on the root process
    if (rank == 0) {
        printf("Inverted image data:\n");
        for (int i = 0; i < total_pixels * 4; ++i) {
            printf("%d ", result[i]);
            if ((i + 1) % 4 == 0) {
                printf("\n");
            }
        }
    }

    // Free memory
    free(local_image);
    if (rank == 0) {
        free(image);
        free(result);
    }

    MPI_Finalize();
    return 0;
}