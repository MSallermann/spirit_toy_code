#ifdef BACKEND_CPU
#include "backend.hpp"
#include "backend_cpu/cpu_helper_functions.hpp"
#include <iostream>

#ifdef USE_OPENMP
#include "omp.h"
#endif

std::string description()
{
    std::string des;
#ifdef USE_OPENMP
    des = "CPU with OpenMP and " + std::to_string( omp_get_max_threads() ) + " thread(s)";
#else
    des = "CPU";
#endif
    return des;
}

void gradient( Backend_Handle & state )
{
    int Na            = state.n_cells[0];
    int Nb            = state.n_cells[1];
    int Nc            = state.n_cells[2];
    int N_cells_total = Na * Nb * Nc;

#pragma omp parallel for
    for( int i = 0; i < state.nos; i++ )
    {
        state.gradient[i] = { 0, 0, 0 };
    }

    #pragma omp parallel for
    for( int i_cell = 0; i_cell < N_cells_total; i_cell++ )
    {
        int tupel[3];
        tupel_from_idx( i_cell, tupel, state.n_cells, 3 );
        int a = tupel[0];
        int b = tupel[1];
        int c = tupel[2];
        for( int i = 0; i < state.n_cell_atoms; i++ )
        {
            int idx_i = i + state.n_cell_atoms * ( i_cell );
            for( int p = 0; p < state.N_pair; p++ )
            {
                const Pair_Stencil & pair = state.pair_stencils[p];
                int idx_j                 = pair.j + state.n_cell_atoms * ( ( a + pair.da ) + Na * ( b + pair.db + Nc * ( c + pair.dc ) ) );
                if( i == pair.i && idx_j > 0 && idx_j < state.nos )
                {
                    state.gradient[idx_i] += pair.matrix * state.spins[idx_j];
                }
            }
        }
    }
}

void propagate_spins( Backend_Handle & state )
{
#pragma omp parallel for
    for( int idx = 0; idx < state.nos; idx++ )
    {
        state.spins[idx] += state.timestep * state.gradient[idx];
        state.spins[idx].normalize();
    }
}

void iterate( State & state, int N_iterations )
{
    for( int iter = 0; iter < N_iterations; iter++ )
    {
        gradient( *state.backend );
        propagate_spins( *state.backend );
        if( iter % state.n_log == 0 )
        {
            printf( "iter = %i\n", iter );
            state.backend->Download( state );
            std::cout << "Spin[0,0,0] = " << state.spins[0].transpose() << "\n";
        }
    }
    state.backend->Download( state );
}

#endif