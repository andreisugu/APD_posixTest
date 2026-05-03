/*
 * ============================================================
 * TEMA LABORATOR: Roy-Floyd-Warshall (C++ Threads)
 * ============================================================
 * Problema: Gasirea distantelor minime intre oricare doua noduri.
 *
 * Strategie de paralelizare:
 * - N=8 noduri, 4 fire de executie (threads).
 * - Algoritmul Roy-Floyd are 3 bucle: for(k), for(i), for(j).
 * - La fiecare pas 'k', paralelizam bucla 'i' (liniile matricei).
 * - Impartim cele N linii in mod egal intre cele NPROCS threads.
 * - Fiecare fir calculeaza o "felie" din matrice.
 * - Folosim metoda join() la finalul fiecarei iteratii 'k' pentru
 * a ne asigura ca toata matricea a fost actualizata inainte 
 * de a trece la urmatorul nod intermediar.
 *
 * Compilare: g++ -std=c++11 -pthread -o roy_floyd_threads roy_floyd_threads.cpp
 * Rulare:    ./roy_floyd_threads
 * ============================================================
 */

#include <iostream>
#include <thread>
#include <vector>
#include <chrono>
#include <cmath>

using namespace std;
using namespace std::chrono;

#define N 8
#define NPROCS 4
#define INF 99999

/*
 * ==========================================================
 * Functia executata de fiecare thread.
 * Calculeaza drumurile minime pentru un set specific de linii
 * (de la start_row pana la end_row - 1).
 * ==========================================================
 */
void compute_row_range(int* D, int start_row, int end_row, int k)
{
    for (int i = start_row; i < end_row; i++)
    {
        for (int j = 0; j < N; j++)
        {
            int dist_ik = D[i * N + k];
            int dist_kj = D[k * N + j];

            /* Evitam overflow la adunarea cu infinit */
            if (dist_ik != INF && dist_kj != INF)
            {
                if (dist_ik + dist_kj < D[i * N + j])
                {
                    D[i * N + j] = dist_ik + dist_kj;
                }
            }
        }
    }
}

int main()
{
    int* D = new int[N * N];

    /* ==========================================================
     * PASUL 1: Initializarea datelor
     * ========================================================== */
    cout << "Matricea initiala a distantelor (INF=" << INF << "):\n";
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
                cout << "  INF";
            else
                printf(" %4d", D[i * N + j]);
        }
        cout << "\n";
    }
    cout << "\n";

    /* ==========================================================
     * PASUL 2: Masurarea timpului (START)
     * Folosim high_resolution_clock din biblioteca <chrono>
     * ========================================================== */
    const auto startTime = high_resolution_clock::now();

    /* ==========================================================
     * PASUL 3: Algoritmul Roy-Floyd-Warshall cu Threads
     * ========================================================== */
    int rows_per_thread = N / NPROCS;

    for (int k = 0; k < N; k++)
    {
        vector<thread> threads;

        /* Lansam firele de executie pentru iteratia curenta k */
        for (int t = 0; t < NPROCS; t++)
        {
            int start_row = t * rows_per_thread;
            /* Ultimul thread ia restul liniilor in caz ca N nu se imparte exact la NPROCS */
            int end_row = (t == NPROCS - 1) ? N : start_row + rows_per_thread;

            /* Initializarea firului cu o functie si argumentele ei */
            threads.push_back(thread(compute_row_range, D, start_row, end_row, k));
        }

        /* Asteptam ca toate firele sa isi termine bucata inainte de a trece la k+1 */
        for (int t = 0; t < NPROCS; t++)
        {
            if (threads[t].joinable())
            {
                threads[t].join();
            }
        }
    }

    /* ==========================================================
     * PASUL 4: Masurarea timpului (STOP)
     * ========================================================== */
    const auto endTime = high_resolution_clock::now();
    double duration_ms = duration_cast<duration<double, milli>>(endTime - startTime).count();

    /* ==========================================================
     * PASUL 5: Afisare rezultate finale
     * ========================================================== */
    cout << "Matricea finala a distantelor minime:\n";
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf(" %4d", D[i * N + j]);
        }
        cout << "\n";
    }

    cout << "\n=== TIMP DE EXECUTIE ===\n";
    cout << "Timp C++ Threads: " << duration_ms / 1000.0 << " secunde\n";

    delete[] D;
    return 0;
}