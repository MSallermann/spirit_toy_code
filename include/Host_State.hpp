#pragma once
#ifndef HOST_STATE_HPP
#define HOST_STATE_HPP
#include "Device_State.hpp"
#include "Hamiltonian.hpp"
#include "definitions.hpp"
#include <array>
#include <memory>
#include <vector>

class Host_State
{
public:
    // Host Stack memory
    int nos;
    int n_cell_atoms;
    int n_cells_total;
    std::array<int, 3> n_cells;
    scalar timestep = 1e-3;

    // Host Heap memory
    std::vector<Vector3> spins;
    std::vector<Vector3> gradient;
    std::vector<ED_Stencil> ed_stencils;

    Device_State device_state;

    Host_State( std::array<int, 3> n_cells, int n_cell_atoms, const std::vector<ED_Stencil> & ed_stencils ) : n_cells( n_cells ), n_cell_atoms( n_cell_atoms ), ed_stencils( ed_stencils )
    {
        timestep      = 1e-3;
        n_cells_total = n_cells[0] * n_cells[1] * n_cells[2];
        nos           = n_cell_atoms * n_cells_total;
        gradient      = std::vector<Vector3>( nos, { 0, 0, 0 } );
        spins         = std::vector<Vector3>( nos, { 1, 1, 1 } );
        Allocate_GPU_Backend();
        Upload();
    }

    void Set_Domain( const Vector3 & vec )
    {
#pragma omp parallel for
        for( int i = 0; i < nos; i++ )
        {
            spins[i] = vec;
        }
        Upload();
    }

    void Download()
    {
        device_state.download( this );
    }

    void Upload()
    {
        device_state.upload( this );
    }

    void Allocate_GPU_Backend()
    {
        device_state.allocate( this );
    }

    ~Host_State()
    {
        device_state.free();
    }
};

#endif