#ifdef BACKEND_CUDA
#pragma once
#ifndef IMPLEMENTATION_STENCIL_EVALUATOR_CUDA_HPP
#define IMPLEMENTATION_STENCIL_EVALUATOR_CUDA_HPP

#include "implementation/State.hpp"
#include "implementation/backend_cuda/cuda_helper_functions.hpp"

namespace Spirit
{
namespace Device
{

template<typename Stencil>
__global__ void __stencil_gradient( Vector3 * gradient, Vector3 * spins, State_Pod state, int N_Stencil, Stencil * stencils )
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int Na = state.n_cells[0];
    int Nb = state.n_cells[1];
    // int Nc = state.n_cells[2]; Not needed

    for( int i_cell = index; i_cell < state.n_cells_total; i_cell += stride )
    {
        int tupel[3];
        CUDA_HELPER::cu_tupel_from_idx( i_cell, tupel, state.n_cells, 3 ); // tupel now is {i, a, b, c}
        int a = tupel[0];
        int b = tupel[1];
        int c = tupel[2];

        for( int i_basis = 0; i_basis < state.n_cell_atoms; i_basis++ )
        {
            int idx_i = i_basis + state.n_cell_atoms * ( i_cell );

            // Allocate data for interacting spins
            Vector3 interaction_spins[Stencil::N_interaction];
            interaction_spins[0] = spins[idx_i];

            for( int p = 0; p < N_Stencil; p++ )
            {
                Stencil & stencil = stencils[p];
                if( stencil.get_i() == i_basis )
                {
                    for( int idx_interaction = 0; idx_interaction < Stencil::N_interaction - 1; idx_interaction++ )
                    {
                        int idx_j = stencil.get_j( idx_interaction )
                                    + state.n_cell_atoms
                                          * ( ( a + stencil.get_da( idx_interaction ) )
                                              + Na * ( b + stencil.get_db( idx_interaction ) + Nb * ( c + stencil.get_dc( idx_interaction ) ) ) );
                        if( idx_j >= 0 && idx_j < state.nos )
                        {
                            interaction_spins[idx_interaction + 1] = spins[idx_j];
                        }
                        else
                        {
                            interaction_spins[idx_interaction + 1] = { 0, 0, 0 };
                        }
                    }
                    gradient[idx_i] += stencil.gradient( interaction_spins );
                }
            }
        }
    }
}

template<typename Stencil>
void stencil_gradient( Vector3 * gradient, Vector3 * spins, State_Pod & state, int N_Stencil, Stencil * stencils )
{
    int blockSize = 1024;
    int numBlocks = ( state.n_cells_total + blockSize - 1 ) / blockSize;
    __stencil_gradient<<<numBlocks, blockSize>>>( gradient, spins, state, N_Stencil, stencils );
}

} // namespace Device
} // namespace Spirit

#endif
#endif