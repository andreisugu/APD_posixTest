# Lesson 1 — MPI Lab (APD_posixTest)

The goals of this lesson are:
- understanding the MPI programming model
- managing the MPI environment
- handling errors
- point-to-point communication

## 1. The MPI Environment
MPI (Message Passing Interface) is a message-passing library standard (see http://www.mpi-forum.org/).
These labs use the MPICH implementation (http://www.mpich.org/).

### 1.1 The MPI programming model
MPI assumes a distributed-memory programming model. Typical program structure:

```c
/* headers */
/* declarations/prototypes */
int main(int argc, char *argv[]) {
  /* serial code */
  MPI_Init(&argc,&argv); /* init MPI */
  /* parallel code: rank/size, send/recv, compute */
  MPI_Finalize(); /* finalize MPI */
  return 0;
}
```

Remember to include the header:

```c
#include "mpi.h"
```

### 1.2 Managing the MPI environment
- `MPI_Init(int *argc, char ***argv)` — initialize MPI (call once, before other MPI calls)
- `MPI_Initialized(int *flag)` — check if MPI was initialized
- `MPI_Finalize()` — finalize MPI
- `MPI_Abort(MPI_Comm comm, int errorcode)` — abort all processes in a communicator

Use `MPI_COMM_WORLD` for the full-job communicator. Get rank/size with:

- `MPI_Comm_rank(MPI_Comm comm, int *rank)`
- `MPI_Comm_size(MPI_Comm comm, int *size)`

Other useful calls: `MPI_Get_processor_name`, `MPI_Get_version`, `MPI_Wtime`.

### 1.3 Error handling
By default MPI aborts on errors (MPI_ERRORS_ARE_FATAL). To return error codes instead, use
`MPI_Comm_set_errhandler(MPI_Comm comm, MPI_Errhandler errhandler)` (for example `MPI_ERRORS_RETURN`).

## 2. Point-to-point communication
Point-to-point is message exchange between two processes. Key primitives:

- Blocking: `MPI_Send`, `MPI_Recv`
- Non-blocking: `MPI_Isend`, `MPI_Irecv` (use `MPI_Request` and `MPI_Wait`/`MPI_Waitall`)
- Synchronous: `MPI_Ssend`

MPI provides buffering so send/receive calls can be asynchronous with respect to application code. Messages
between a pair of processes are delivered in send order.

## 3. Examples and lab files (in this repo)
- [example1.c](example1.c) — simple MPI example
- [example2.c](example2.c) — additional example
- [Lab01/mpi_test.c](Lab01/mpi_test.c) — lab 1 test program
- [mpi_test.py](mpi_test.py) — helper/test script

Build/run notes:

```bash
mpicc Lab01/mpi_test.c -o mpi_test
mpirun -np 4 ./mpi_test
```

## 4. Exercises
1. Write a program that prints all prime numbers less than N using M processes.
2. Write a program that searches an element inside an array and prints its position or `Not found`.
3. For n processes: each generates m (m>=100) random numbers (<=1000), prints them and their sum; measure per-process time.

---
Focus: these materials are the course's first lab; for our project use the `Lab01` and example files as starting points.
