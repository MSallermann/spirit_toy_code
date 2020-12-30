#ifdef BACKEND_CPU

#pragma once
#ifndef IMPLEMENTATION_STENCIL_EVALUATOR_CPU_HPP
#define IMPLEMENTATION_STENCIL_EVALUATOR_CPU_HPP

#include "implementation/Fields.hpp"
#include "implementation/backend_cpu/cpu_helper_functions.hpp"
#include "interface/State.hpp"

namespace Spirit
{
namespace Implementation
{

template<typename Stencil>
void stencil_gradient( Vector3 * gradient, Vector3 * spins, Interface::State::Geometry & geometry, int N_Stencil, Stencil * stencils )
{
    int Na = geometry.n_cells[0];
    int Nb = geometry.n_cells[1];
    int Nc = geometry.n_cells[2];

#pragma omp parallel for
    for( int i_cell = 0; i_cell < geometry.n_cells_total; i_cell++ )
    {
        int tupel[3];
        CPU_HELPER::tupel_from_idx( i_cell, tupel, geometry.n_cells, 3 ); // tupel now is {i, a, b, c}
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

} // namespace Implementation
} // namespace Spirit

#endif

#endif
