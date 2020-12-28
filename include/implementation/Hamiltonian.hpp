#pragma once
#ifndef IMPLEMENTATION_HAMILTONIAN_HPP
#define IMPLEMENTATION_HAMILTONIAN_HPP
#include "implementation/Memory.hpp"
#include "implementation/Stencil_Terms.hpp"
#include "interface/Hamiltonian.hpp"

#include <array>
namespace Spirit
{
namespace Implementation
{

class Hamiltonian
{
public:
    Hamiltonian( Spirit::Interface::Hamiltonian * ham )
            : boundary_conditions( ham->boundary_conditions ),
              ed_stencils( device_vector<ED_Stencil>( ham->ed_stencils.size() ) ),
              k_stencils( device_vector<K_Stencil>( ham->k_stencils.size() ) ),
              b_stencils( device_vector<Bfield_Stencil>( ham->b_stencils.size() ) )
    {

        auto ptr_ed = (ED_Stencil *)( ham->ed_stencils.data() );
        ed_stencils.copy_from( ptr_ed );

        auto ptr_k = (K_Stencil *)( ham->k_stencils.data() );
        k_stencils.copy_from( ptr_k );

        auto ptr_b = (Bfield_Stencil *)( ham->b_stencils.data() );
        b_stencils.copy_from( ptr_b );
    }

    std::array<bool, 3> boundary_conditions;
    device_vector<ED_Stencil> ed_stencils;
    device_vector<K_Stencil> k_stencils;
    device_vector<Bfield_Stencil> b_stencils;
};

} // namespace Implementation
} // namespace Spirit

#endif