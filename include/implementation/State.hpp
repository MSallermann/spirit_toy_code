#pragma once
#ifndef IMPLEMENTATION_STATE_HPP
#define IMPLEMENTATION_STATE_HPP
#include "Definitions.hpp"
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
            : n_cells( state->n_cells ),
              nos( state->nos ),
              n_cell_atoms( state->n_cell_atoms ),
              n_cells_total( state->n_cells_total ),
              spins( device_vector<Vector3>( state->nos ) ),
              gradient( device_vector<Vector3>( state->nos ) )
    {
    }

    std::array<int, 3> n_cells;
    int nos;
    int n_cell_atoms;
    int n_cells_total;
    scalar timestep;

    device_vector<Vector3> spins;
    device_vector<Vector3> gradient;
};

} // namespace Device
} // namespace Spirit

#endif