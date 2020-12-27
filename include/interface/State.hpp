#pragma once
#ifndef INTERFACE_STATE_HPP
#define INTERFACE_STATE_HPP
#include "Definitions.hpp"
#include "interface/Hamiltonian.hpp"

#include <array>
#include <vector>

namespace Spirit
{

namespace Device
{
class State;
}

namespace Host
{
class State
{

protected:
    Device::State * device_state = nullptr;

public:
    State( std::array<int, 3> n_cells, int n_cell_atoms ) : n_cells( n_cells ), n_cell_atoms( n_cell_atoms )
    {
        n_cells_total = n_cells[0] * n_cells[1] * n_cells[2];
        nos           = n_cells_total * n_cell_atoms;
    }

    std::array<int, 3> n_cells;
    int n_cell_atoms;
    int n_cells_total;
    int nos;
    scalar timestep = 1e-3;

    // Host Heap memory
    std::vector<Vector3> spins;
    std::vector<Vector3> gradient;

    Hamiltonian hamiltonian;

    void allocate();
    void upload();
    void download();
    ~State();
};

} // namespace Host
} // namespace Spirit

#endif