#include "State.hpp"
#include "backend.hpp"

State::State( std::array<int, 3> n_cells, int n_cell_atoms, scalar timestep ) : n_cells( n_cells ), n_cell_atoms( n_cell_atoms ), timestep( timestep )
{
    this->spins = std::vector<Vector3>( Nos(), { 0, 1, 1 } );
    Create_Backend();
}

void State::Set_Pair_Stencils( std::vector<Pair_Stencil> & stencils )
{
    this->pair_stencils = stencils;
    Create_Backend();
}