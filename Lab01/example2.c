#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#define MASTER 0
int main(int argc, char *argv[])
{
    int numprocs, procid, len;
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int partner, message;
    // MPI_Status status; OLD BAD
    MPI_Status statuses[2];
    MPI_Request reqs[2];
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);
    MPI_Get_processor_name(hostname, &len);
    printf("Hello from proc %d on %s!\n", procid, hostname);
    if (procid == MASTER)
        printf("MASTER: Number of MPI procs is: %d\n", numprocs);
    /* determine partner and then send/receive with partner */
    if (procid < numprocs / 2)
        partner = numprocs / 2 + procid;
    else if (procid >= numprocs / 2)
        partner = procid - numprocs / 2;
    MPI_Irecv(&message, 1, MPI_INT, partner, 1, MPI_COMM_WORLD, &reqs[0]);
    MPI_Isend(&procid, 1, MPI_INT, partner, 1, MPI_COMM_WORLD, &reqs[1]);
    /* now block until requests are complete */
    // MPI_Waitall(2, reqs, &status); OLD BAD
    MPI_Waitall(2, reqs, statuses);
    /* print partner info and exit*/
    printf("Proc %d is partner with %d\n", procid, message);
    MPI_Finalize();
}