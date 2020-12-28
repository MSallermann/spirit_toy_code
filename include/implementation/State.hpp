#pragma once
#ifndef IMPLEMENTATION_STATE_HPP
#define IMPLEMENTATION_STATE_HPP
#include "Definitions.hpp"
#include "implementation/Hamiltonian.hpp"
#include "implementation/Memory.hpp"
#include "interface/State.hpp"

namespace Spirit
{
namespace Implementation
{

struct State_Pod
{
    int n_cells[3];
    int nos;
    int n_cell_atoms;
    int n_cells_total;
    scalar timestep;
};

class State
{
public:
    State( Spirit::Interface::State * state )
            : spins( device_vector<Vector3>( state->nos ) ),
              gradient( device_vector<Vector3>( state->nos ) ),
              hamiltonian( Hamiltonian( &state->hamiltonian ) )
    {
        pod.n_cells[0]    = state->n_cells[0];
        pod.n_cells[1]    = state->n_cells[1];
        pod.n_cells[2]    = state->n_cells[2];
        pod.nos           = state->nos;
        pod.n_cell_atoms  = state->n_cell_atoms;
        pod.n_cells_total = state->n_cells_total;
        pod.timestep      = state->timestep;
    }

    device_vector<Vector3> spins;
    device_vector<Vector3> gradient;
    State_Pod pod;
    Hamiltonian hamiltonian;

    State * device_handle;
    void get_gradient( Vector3 * gradient, Vector3 * spins, State_Pod & state_pod );
};

} // namespace Implementation
} // namespace Spirit

#endif