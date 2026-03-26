#include "mpi.h"
#include <stdio.h>
#include <string.h>

#define NUM_STUDENTS 8
#define NAME_LEN 32

typedef struct {
    int id;
    char name[NAME_LEN];
    float grade;
} Student;

int main(int argc, char* argv[]) {
    int rank, numtasks, i;
    Student students[NUM_STUDENTS];
    MPI_Datatype studenttype, oldtypes[3];
    int blockcounts[3];
    MPI_Aint offsets[3], lb, extent;
    int search_id = 5; /* ID-ul studentului cautat de toate procesele */

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    /* -------------------------------------------------------
       Construim tipul derivat MPI_Struct pentru Student.
       Structura are 3 campuri: int id | char name[32] | float grade
       Trebuie sa specificam: tipul, numarul de elemente si
       offset-ul (in bytes) al fiecarui camp in struct.
    ------------------------------------------------------- */

    /* Campul 0: int id, incepe la offset 0 */
    offsets[0]    = 0;
    oldtypes[0]   = MPI_INT;
    blockcounts[0]= 1;

    /* Campul 1: char name[NAME_LEN], incepe dupa un int */
    MPI_Type_get_extent(MPI_INT, &lb, &extent); /* extent = sizeof(int) */
    offsets[1]    = extent;
    oldtypes[1]   = MPI_CHAR;
    blockcounts[1]= NAME_LEN;

    /* Campul 2: float grade, incepe dupa int + NAME_LEN chars */
    MPI_Type_get_extent(MPI_CHAR, &lb, &extent); /* extent = sizeof(char) = 1 */
    offsets[2]    = offsets[1] + NAME_LEN * extent;
    oldtypes[2]   = MPI_FLOAT;
    blockcounts[2]= 1;

    /* Cream si inregistram tipul struct */
    MPI_Type_create_struct(3, blockcounts, offsets, oldtypes, &studenttype);
    MPI_Type_commit(&studenttype);

    /* -------------------------------------------------------
       Procesul 0 initializeaza lista de studenti.
    ------------------------------------------------------- */
    if (rank == 0) {
        for (i = 0; i < NUM_STUDENTS; i++) {
            students[i].id = i + 1;
            snprintf(students[i].name, NAME_LEN, "Student_%d", i + 1);
            // snprintf = format string in students[i].name with "Student_%d" where %d is i+1
            students[i].grade = 5.0f + i * 0.5f;
        }
    }

    /* -------------------------------------------------------
       Broadcast: procesul 0 trimite intreaga lista catre
       toate procesele. Dupa Bcast, fiecare proces are
       o copie completa a tabloului students[].
    ------------------------------------------------------- */
    MPI_Bcast(students, NUM_STUDENTS, studenttype, 0, MPI_COMM_WORLD);

    /* -------------------------------------------------------
       Fiecare proces cauta in propria sa portiune din lista.
       Impartim lista in chunk-uri egale (restul merge la ultimul).
    ------------------------------------------------------- */
    int chunk = NUM_STUDENTS / numtasks;
    int start = rank * chunk;
    int end   = (rank == numtasks - 1) ? NUM_STUDENTS : start + chunk;

    for (i = start; i < end; i++) {
        if (students[i].id == search_id) {
            printf("Procesul %d a gasit: ID=%d, Nume=%s, Nota=%.1f\n",
                   rank, students[i].id, students[i].name, students[i].grade);
        }
    }

    /* Daca procesul nu a gasit nimic, nu afiseaza nimic */

    MPI_Type_free(&studenttype);
    MPI_Finalize();
    return 0;
}
