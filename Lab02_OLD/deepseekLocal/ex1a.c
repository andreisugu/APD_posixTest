#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>  // Include this header for malloc

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <array_size> [target] [array_elements...]\n", argv[0]);
        return 1;
    }

    int array_size = argc > 1 ? atoi(argv[1]) : 5;
    int target = argc > 2 ? atoi(argv[2]) : -1;
    int *array = NULL;

    if (argc > 3) {
        if (argc - 3 != array_size) {
            printf("Error: Number of array elements must match array size\n");
            return 1;
        }
        array = malloc(array_size * sizeof(int));
        for (int i = 0; i < array_size; i++) {
            array[i] = atoi(argv[3 + i]);
        }
    } else {
        array = malloc(array_size * sizeof(int));
        for (int i = 0; i < array_size; i++) {
            array[i] = i + 1;
        }
    }

    if (argc < 2) {
        printf("Usage: %s <array_size> [target] [array_elements...]\n", argv[0]);
        return 1;
    }

    int array_size = argc > 1 ? atoi(argv[1]) : 5;
    int target = argc > 2 ? atoi(argv[2]) : -1;
    int *array = NULL;

    if (argc > 3) {
        if (argc - 3 != array_size) {
            printf("Error: Number of array elements must match array size\n");
            return 1;
        }
        array = malloc(array_size * sizeof(int));
        for (int i = 0; i < array_size; i++) {
            array[i] = atoi(argv[3 + i]);
        }
    } else {
        array = malloc(array_size * sizeof(int));
        for (int i = 0; i < array_size; i++) {
            array[i] = i + 1;
        }
    }
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int target;
    int array_size;
    int *array;

    if (rank == 0) {
        // Read input array and target element
        printf("Enter array size: ");
        if (scanf("%d", &array_size) != 1) {
            fprintf(stderr, "Error reading array size\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        
        if (array_size <= 0) {
            fprintf(stderr, "Array size must be positive\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("Enter target element: ");
        if (scanf("%d", &target) != 1) {
            fprintf(stderr, "Error reading target element\n");
            free(array);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        array = malloc(array_size * sizeof(int));
        if (array == NULL) {
            fprintf(stderr, "Memory allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        printf("Enter array elements: ");
        for (int i = 0; i < array_size; i++) {
            if (scanf("%d", &array[i]) != 1) {
                fprintf(stderr, "Error reading array element at index %d\n", i);
                free(array);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }
    }

    // Broadcast target to all processes
    MPI_Bcast(&target, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Broadcast array size to all processes
    MPI_Bcast(&array_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // Broadcast the array to all processes
    MPI_Bcast(array, array_size, MPI_INT, 0, MPI_COMM_WORLD);

    int chunk_size = (array_size + size - 1) / size; // Calculate chunk size to handle all elements
    int start = rank * chunk_size;
    int max_pos = -1;

    // Search for the target in the local chunk
    for (int i = start; i < start + chunk_size; i++) {
        if (i >= array_size) break;
        if (array[i] == target) {
            if (i > max_pos) {
                max_pos = i;
            }
        }
    }

    // Reduce to find the global maximum position
    int global_max;
    MPI_Reduce(&max_pos, &global_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        if (global_max != -1) {
            printf("Element found at position: %d\n", global_max);
        } else {
            printf("Element not found in the array.\n");
        }
    }

    // Free the allocated memory for the array on the root process
    if (rank == 0) {
        free(array);
    }

    MPI_Finalize();
    return 0;
}