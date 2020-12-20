#ifdef BACKEND_CPU
#include "backend_handle.hpp"
#include "State.hpp"
#include <iostream>

Backend_Handle::Backend_Handle(State & state) 
{
    this->n_cells       = state.n_cells.data();
    this->n_cell_atoms  = state.n_cell_atoms;
    this->spins         = state.spins.data();
    this->pair_stencils = state.pair_stencils.data();
    this->N_pair        = state.pair_stencils.size();
    this->timestep      = state.timestep;
    this->nos           = state.Nos();
    this->gradient = new Vector3[state.Nos()];
}

void Backend_Handle::Upload( State & state )
{
    // Nothing to be done
}

void Backend_Handle::Download( State & state )
{
    // Nothing to be done
}

Backend_Handle::~Backend_Handle()
{
    std::cout << "Destructor Backend Handle\n";
    delete[] gradient;
}

#endif