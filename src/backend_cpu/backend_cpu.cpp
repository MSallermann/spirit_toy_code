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

void set_gradient_zero( Device_State state )
{
#pragma omp parallel for
    for( int i = 0; i < state.nos; i++ )
    {
        state.gradient[i] = { 0, 0, 0 };
    }
}

template<int N, typename Stencil>
void stencil_gradient( Device_State state )
{
    int Na = state.n_cells[0];
    int Nb = state.n_cells[1];
    // int Nc = state.n_cells[2]; Not needed

    auto N_Stencil = get_n_stencil<Stencil>( state );
    auto stencils  = get_stencils<Stencil>( state );

#pragma omp parallel for
    for( int i_cell = 0; i_cell < state.n_cells_total; i_cell++ )
    {
        int tupel[3];
        CPU_HELPER::tupel_from_idx( i_cell, tupel, state.n_cells, 3 ); // tupel now is {i, a, b, c}
        int a = tupel[0];
        int b = tupel[1];
        int c = tupel[2];

        for( int i_basis = 0; i_basis < state.n_cell_atoms; i_basis++ )
        {
            int idx_i = i_basis + state.n_cell_atoms * ( i_cell );

            // Allocate data for interacting spins
            Vector3 interaction_spins[N];

            for( int p = 0; p < N_Stencil; p++ )
            {
                Stencil & stencil = stencils[p];
                if( stencil.get_i() == i_basis )
                {
                    for( int idx_interaction = 0; idx_interaction < N; idx_interaction++ )
                    {
                        int idx_j
                            = stencil.get_j( idx_interaction )
                              + state.n_cell_atoms * ( ( a + stencil.get_da( idx_interaction ) ) + Na * ( b + stencil.get_db( idx_interaction ) + Nb * ( c + stencil.get_dc( idx_interaction ) ) ) );
                        if( idx_j >= 0 && idx_j < state.nos )
                        {
                            interaction_spins[idx_interaction] = state.spins[idx_j];
                        }
                        else
                        {
                            interaction_spins[idx_interaction] = { 0, 0, 0 };
                        }
                    }
                    state.gradient[idx_i] += stencil.gradient( state.spins[idx_i], interaction_spins );
                }
            }
        }
    }
}

void propagate_spins( Device_State state )
{
#pragma omp parallel for
    for( int idx = 0; idx < state.nos; idx++ )
    {
        state.spins[idx] += state.timestep * state.gradient[idx];
        state.spins[idx].normalize();
    }
}

void iterate( Host_State & state, int N_iterations )
{
    for( int iter = 0; iter < N_iterations; iter++ )
    {
        set_gradient_zero( state.device_state );
        stencil_gradient<1, ED_Stencil>( state.device_state );

        propagate_spins( state.device_state );
        if( iter % 250 == 0 )
        {
            printf( "iter = %i\n", iter );
            state.Download();
            std::cout << "Spin[0,0,0] = " << state.spins[0].transpose() << "\n";
        }
    }
    state.Download();
}

#endif