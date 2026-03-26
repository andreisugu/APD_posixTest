Iată cele 3 exerciții, fiecare cu comentarii detaliate. Un rezumat rapid al logicii din fiecare:

---

**Ex1 — Student (Struct)**
- `MPI_Type_create_struct` cu 3 câmpuri: `int id`, `char name[32]`, `float grade`
- Offset-urile sunt calculate cu `MPI_Type_get_extent` (exact ca în Exemplul 4 din lab)
- Procesul 0 inițializează lista → `MPI_Bcast` trimite tot la toți → fiecare proces caută în **chunk-ul său** (`rank * chunk` până la `(rank+1) * chunk`)

---

**Ex2 — Diagonale (Indexed)**
- Matricea e stocată **row-major** (plat): `a[i][j]` = `a_flat[i*N + j]`
- Diagonala principală: indici `i*(N+1)` → pentru N=4: `0, 5, 10, 15`
- Diagonala secundară: indici `i*N + (N-1-i)` → pentru N=4: `3, 6, 9, 12`
- `MPI_Type_indexed` primește ambele seturi (2*N blocuri de câte 1 element), extrage valorile și le trimite **contiguu** în buffer-ul receptorului

---

**Ex3 — Coloane consecutive (Vector)**
- `MPI_Type_vector(ROWS, 2, COLS, ...)` descrie 2 coloane consecutive: `blocklength=2` (2 elemente per rând), `stride=COLS` (salt la rândul următor)
- Procesul `dest` primește coloanele `2*(dest-1)` și `2*(dest-1)+1`; trimiterea pornește din `&a[0][col_start]`
- Receptorul primește datele **contiguu** în `b[]` (rând cu rând) și calculează sumă, medie, maxim

**Compilare:**
```bash
mpicc ex1_student.c -o ex1 && mpirun -np 4 ./ex1
mpicc ex2_diagonals.c -o ex2 && mpirun -np 3 ./ex2
mpicc ex3_columns.c -o ex3 && mpirun -np 5 ./ex3
```