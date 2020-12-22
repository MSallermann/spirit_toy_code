#ifdef BACKEND_CPU

#include "Device_State.hpp"
#include "Host_State.hpp"

void Device_State::allocate( Host_State * host_state )
{
    this->free();
    allocated = true;

    this->allocated     = true;
    this->nos           = host_state->nos;
    this->n_cells_total = host_state->n_cells_total;
    this->n_cell_atoms  = host_state->n_cell_atoms;
    this->n_cells[0]    = host_state->n_cells[0];
    this->n_cells[1]    = host_state->n_cells[1];
    this->n_cells[2]    = host_state->n_cells[2];
    this->n_ed          = host_state->ed_stencils.size();
    this->timestep      = host_state->timestep;

    this->gradient    = host_state->gradient.data();
    this->spins       = host_state->spins.data();
    this->ed_stencils = host_state->ed_stencils.data();
}

void Device_State::download( Host_State * host_state )
{
    return;
}

void Device_State::upload( Host_State * host_state )
{
    return;
}

void Device_State::free()
{
    return;
}

#endif