#ifndef STATE_HPP
#define STATE_HPP
#include "backend_handle.hpp"
#include "definitions.hpp"
#include <array>
#include <vector>

class State
{
public:
    State( std::array<int, 3> n_cells, int n_cell_atoms, scalar timestep );
    void Set_Pair_Stencils( std::vector<Pair_Stencil> & stencils );

    int Nos()
    {
        return n_cell_atoms * n_cells[0] * n_cells[1] * n_cells[2];
    }

    Backend_Handle backend;
    std::array<int, 3> n_cells;
    int n_cell_atoms;
    scalar timestep;
    std::vector<Vector3> spins;
    std::vector<Pair_Stencil> pair_stencils;
};

#endif