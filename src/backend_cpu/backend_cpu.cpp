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

void create_backend_handle( State & state )
{
    Backend_Handle & res = state.backend;
    res.n_cells          = state.n_cells.data();
    res.n_cell_atoms     = state.n_cell_atoms;
    res.spins            = state.spins.data();
    res.pair_stencils    = state.pair_stencils.data();
    res.N_pair           = state.pair_stencils.size();
    res.timestep         = state.timestep;
    res.nos              = state.Nos();
    res.gradient         = new Vector3[state.Nos()];
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

    int tupel[3];
#pragma omp parallel for private( tupel )
    for( int i_cell = 0; i_cell < N_cells_total; i_cell++ )
    {
        tupel_from_idx( i_cell, tupel, state.n_cells, 3 );
        int a = tupel[0], b = tupel[1], c = tupel[2];
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

void iterate( Backend_Handle & state, int N_iterations )
{
    for( int iter = 0; iter < N_iterations; iter++ )
    {
        gradient( state );
        propagate_spins( state );
    }
}

#endif