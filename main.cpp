#include <iostream>
#include <chrono>
#include "backend.hpp"

constexpr int N_iterations = 800;

int main()
{
    State state = State({100, 100, 1}, 1, 1e-3);
    
    std::vector<Pair_Stencil> stencils;

    
    Pair_Stencil temp; 
    temp.i = 0;
    temp.j = 0;
    temp.da = 1;
    temp.db = 2;
    temp.dc = 3;
    temp.matrix = 1 * Matrix3::Identity();

    for(int i=0; i<10; i++)
        stencils.push_back( temp );

    state.Set_Pair_Stencils(stencils);
    state.Create_Backend_Handle();

    auto start = std::chrono::high_resolution_clock::now();
    iterate(state.backend, N_iterations);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds = end-start;

    std::cout << "Elapsed time: " << elapsed_seconds.count() * 1e3 << " ms (" << N_iterations << " iterations)\n";
    std::cout << "              " << elapsed_seconds.count() / N_iterations * 1e3 << " ms per iteration\n";
    std::cout << "              " <<  N_iterations / elapsed_seconds.count() << " iterations per second\n";
    
}