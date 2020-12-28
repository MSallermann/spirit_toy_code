#pragma once
#ifndef IMPLEMENTATION_STATE_HPP
#define IMPLEMENTATION_STATE_HPP
#include "Definitions.hpp"
#include "implementation/Hamiltonian.hpp"
#include "implementation/Memory.hpp"
#include "interface/State.hpp"

namespace Spirit
{
namespace Device
{

class State
{
public:
    State( Spirit::Host::State * state )
            : nos( state->nos ),
              n_cell_atoms( state->n_cell_atoms ),
              n_cells_total( state->n_cells_total ),
              timestep( state->timestep ),
              spins( device_vector<Vector3>( state->nos ) ),
              gradient( device_vector<Vector3>( state->nos ) ),
              hamiltonian( Hamiltonian( &state->hamiltonian ) )
    {
        n_cells[0] = state->n_cells[0];
        n_cells[1] = state->n_cells[1];
        n_cells[2] = state->n_cells[2];
    }

    int n_cells[3];
    int nos;
    int n_cell_atoms;
    int n_cells_total;
    scalar timestep;

    device_vector<Vector3> spins;
    device_vector<Vector3> gradient;

    Hamiltonian hamiltonian;

    void Gradient_Async( device_vector<Vector3> & gradient );
};

} // namespace Device
} // namespace Spirit

#endif