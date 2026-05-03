/*
 * ============================================================
 * TEMA LABORATOR: Roy-Floyd-Warshall (C++ STL Parallel)
 * ============================================================
 * Problema: Gasirea distantelor minime intre oricare doua noduri.
 *
 * Strategie de paralelizare:
 * - Folosim standardul C++17 si algoritmii din <algorithm> si <execution>.
 * - Cream un vector de indici pentru linii (0, 1, 2 ... N-1).
 * - La fiecare pas 'k', aplicam std::for_each in mod paralel 
 * (std::execution::par) pe acest vector de indici.
 * - Fiecare index 'i' procesat va actualiza toata linia 'i' din matrice.
 * - Sincronizarea este gestionata automat de std::for_each, care
 * blocheaza executia pana la finalizarea tuturor elementelor din range.
 *
 * Compilare (necesita TBB pe unele sisteme): 
 * g++ -std=c++17 -O3 -o roy_floyd_stl roy_floyd_stl.cpp -ltbb
 * Rulare:    ./roy_floyd_stl
 * ============================================================
 */

#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <execution>
#include <numeric>

using namespace std;
using namespace std::chrono;

#define N 8
#define INF 99999

int main()
{
    vector<int> D(N * N);

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
     * PASUL 2: Pregatirea indicilor pentru paralelizare STL
     * Construim un vector {0, 1, 2, ..., N-1} ce reprezinta
     * liniile matricei care vor fi iterare in paralel.
     * ========================================================== */
    vector<int> row_indices(N);
    iota(row_indices.begin(), row_indices.end(), 0);

    /* ==========================================================
     * PASUL 3: Masurarea timpului (START)
     * ========================================================== */
    const auto startTime = high_resolution_clock::now();

    /* ==========================================================
     * PASUL 4: Algoritmul Roy-Floyd-Warshall cu STL Paralel
     * Adaugam std::execution::par ca prim parametru la apelul 
     * algoritmului for_each pentru a paralelizare munca.
     * ========================================================== */
    for (int k = 0; k < N; k++)
    {
        for_each(execution::par, row_indices.begin(), row_indices.end(), [&](int i) 
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
        });
    }

    /* ==========================================================
     * PASUL 5: Masurarea timpului (STOP)
     * ========================================================== */
    const auto endTime = high_resolution_clock::now();
    double duration_ms = duration_cast<duration<double, milli>>(endTime - startTime).count();

    /* ==========================================================
     * PASUL 6: Afisare rezultate finale
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
    cout << "Timp C++ STL Paralel: " << duration_ms / 1000.0 << " secunde\n";

    return 0;
}