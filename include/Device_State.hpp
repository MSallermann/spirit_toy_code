#pragma once
#ifndef DEVICE_STATE_HPP
#define DEVICE_STATE_HPP
#include "Hamiltonian.hpp"
#include "definitions.hpp"

class Host_State;

struct Device_State
{
    bool allocated = false;
    int nos;
    int n_cell_atoms;
    int n_cells_total;
    int n_cells[3];
    scalar timestep;

    Vector3 * spins;
    Vector3 * gradient;

    int n_ed;
    ED_Stencil * ed_stencils;

    void allocate( Host_State * host_state );
    void upload( Host_State * host_state );
    void download( Host_State * host_state );
    void free();
};

#endif