#ifndef STATE_HPP
#define STATE_HPP
#include "backend_handle.hpp"
#include "definitions.hpp"
#include <array>
#include <memory>
#include <vector>

class State
{
public:
    State( std::array<int, 3> n_cells, int n_cell_atoms, scalar timestep );
    void Set_Pair_Stencils( std::vector<Pair_Stencil> & stencils );

    int Nos() const
    {
        return n_cell_atoms * n_cells[0] * n_cells[1] * n_cells[2];
    }

    void Set_Domain( const Vector3 & vec )
    {
#pragma omp parallel for
        for( int i = 0; i < this->Nos(); i++ )
        {
            this->spins[i] = vec;
        }
        this->backend->Upload( *this );
    }

    void Create_Backend()
    {
        this->backend = std::unique_ptr<Backend_Handle>( new Backend_Handle( *this ) );
    }

    std::unique_ptr<Backend_Handle> backend;
    std::array<int, 3> n_cells;
    int n_cell_atoms;
    int n_log = 250;
    scalar timestep;
    std::vector<Vector3> spins;
    std::vector<Pair_Stencil> pair_stencils;
};

#endif