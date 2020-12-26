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
    std::array<int, 3> n_cells;
    int nos;
    int n_cell_atoms;
    int n_cells_total;
    scalar timestep;

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