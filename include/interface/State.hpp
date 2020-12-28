#pragma once
#ifndef INTERFACE_STATE_HPP
#define INTERFACE_STATE_HPP
#include "Definitions.hpp"
#include "interface/Hamiltonian.hpp"

#include <array>
#include <vector>

namespace Spirit
{

namespace Implementation
{
class Fields;
class Hamiltonian;
} // namespace Implementation

namespace Interface
{
class State
{

public:
    struct Geometry
    {
        int n_cells[3];
        int n_cell_atoms;
        int nos;
        int n_cells_total;
    };

    struct Solver_Parameters
    {
        scalar timestep;
    };

    Implementation::Fields * fields                  = nullptr;
    Implementation::Hamiltonian * hamiltonian_device = nullptr;

    State( std::array<int, 3> n_cells, int n_cell_atoms )
    {
        // Initialize geometry
        geometry.n_cells[0]    = n_cells[0];
        geometry.n_cells[1]    = n_cells[1];
        geometry.n_cells[2]    = n_cells[2];
        geometry.n_cell_atoms  = n_cell_atoms;
        geometry.n_cells_total = n_cells[0] * n_cells[1] * n_cells[2];
        geometry.nos           = n_cell_atoms * geometry.n_cells_total;

        // Initialize solver parameters
        solver_parameters.timestep = 1e-3;

        // Initialize host side fields
        spins    = std::vector<Vector3>( geometry.nos );
        gradient = std::vector<Vector3>( geometry.nos );
    }

    Geometry geometry;
    Solver_Parameters solver_parameters;
    Hamiltonian hamiltonian;

    // Host Heap memory
    std::vector<Vector3> spins;
    std::vector<Vector3> gradient;

    void set_domain( const Vector3 & vec )
    {
#pragma omp parallel for
        for( int i = 0; i < geometry.nos; i++ )
        {
            spins[i] = vec;
        }
    }

    void allocate();
    void upload();
    void download();
    ~State();
};

} // namespace Interface
} // namespace Spirit

#endif