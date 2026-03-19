# Lab 1 - Examples

## Exercise 1: Prime Numbers

Find all primes less than N using M processes.

```bash
# Example 1: N=50, 2 processes
mpirun -n 2 ./Lab1Ex1 50

# Example 2: N=100, 4 processes
mpirun -n 4 ./Lab1Ex1 100

# Example 3: N=200, 3 processes
mpirun -n 3 ./Lab1Ex1 200
```

---

## Exercise 2: Array Search

Search for element in array of size S using P processes.

```bash
# Example 1: size=15, search for 42 (not found), 2 processes
mpirun -n 2 ./Lab1Ex2 15 42

# Example 2: size=20, search for 50 (found), 3 processes
mpirun -n 3 ./Lab1Ex2 20 50

# Example 3: size=30, random search, 4 processes
mpirun -n 4 ./Lab1Ex2 30

# Example 4: size=100, random search, 5 processes
mpirun -n 5 ./Lab1Ex2 100
```

---

## Exercise 3: Random Numbers & Timing

Generate M random numbers in each process and measure execution time.

```bash
# Example 1: m=100, 2 processes
mpirun -n 2 ./Lab1Ex3 100

# Example 2: m=200, 3 processes
mpirun -n 3 ./Lab1Ex3 200

# Example 3: m=150, 4 processes
mpirun -n 4 ./Lab1Ex3 150
```

---

## Quick Commands

```bash
# Compile all
mpicc Lab1Ex1.c -o Lab1Ex1
mpicc Lab1Ex2.c -o Lab1Ex2
mpicc Lab1Ex3.c -o Lab1Ex3

# Run all examples
mpirun -n 2 ./Lab1Ex1 50 && \
mpirun -n 3 ./Lab1Ex2 20 50 && \
mpirun -n 2 ./Lab1Ex3 100
```
