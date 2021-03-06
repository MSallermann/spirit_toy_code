#ifdef BACKEND_CUDA
#pragma once
#ifndef IMPLEMENTATION_STENCIL_EVALUATOR_CUDA_HPP
#define IMPLEMENTATION_STENCIL_EVALUATOR_CUDA_HPP
#include "implementation/Fields.hpp"
#include "implementation/backend_cuda/cuda_helper_functions.hpp"
#include "interface/State.hpp"

namespace Spirit
{
namespace Implementation
{

template<typename Stencil>
__global__ void __stencil_gradient( Vector3 * gradient, Vector3 * spins, Interface::State::Geometry geometry, int N_Stencil, Stencil * stencils )
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int Na = geometry.n_cells[0];
    int Nb = geometry.n_cells[1];
    int Nc = geometry.n_cells[2];

    for( int i_cell = index; i_cell < geometry.n_cells_total; i_cell += stride )
    {
        int tupel[3];
        CUDA_HELPER::cu_tupel_from_idx( i_cell, tupel, geometry.n_cells, 3 ); // tupel now is {i, a, b, c}
        int a = tupel[0];
        int b = tupel[1];
        int c = tupel[2];

        for( int i_basis = 0; i_basis < geometry.n_cell_atoms; i_basis++ )
        {
            int idx_i = i_basis + geometry.n_cell_atoms * ( i_cell );

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
                        // Evaluate boundary conditions
                        int new_a = ( a + stencil.get_da( idx_interaction ) );
                        int new_b = ( b + stencil.get_db( idx_interaction ) );
                        int new_c = ( c + stencil.get_dc( idx_interaction ) );
                        int j     = stencil.get_j( idx_interaction );

                        if( new_a < 0 || new_a >= Na )
                            new_a = ( geometry.boundary_conditions[0] ) ? new_a % Na : -2 * geometry.nos;
                        if( new_b < 0 || new_b >= Nb )
                            new_b = ( geometry.boundary_conditions[1] ) ? new_b % Nb : -2 * geometry.nos;
                        if( new_c < 0 || new_c >= Nc )
                            new_c = ( geometry.boundary_conditions[2] ) ? new_c % Nc : -2 * geometry.nos;

                        int idx_j = j + geometry.n_cell_atoms * ( new_a + Na * ( new_b + Nb * ( new_c ) ) );

                        if( idx_j >= 0 && idx_j < geometry.nos )
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
void stencil_gradient( Vector3 * gradient, Vector3 * spins, Interface::State::Geometry & geometry, int N_Stencil, Stencil * stencils )
{
    int blockSize = 1024;
    int numBlocks = ( geometry.n_cells_total + blockSize - 1 ) / blockSize;
    __stencil_gradient<<<numBlocks, blockSize>>>( gradient, spins, geometry, N_Stencil, stencils );
}

} // namespace Implementation
} // namespace Spirit

#endif
#endif