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

    void H_ATTRIBUTE allocate( Host_State * host_state );
    void H_ATTRIBUTE upload( Host_State * host_state );
    void H_ATTRIBUTE download( Host_State * host_state );
    void H_ATTRIBUTE free();
};

// Functions to read Stencils from Device state (They are not defined as class members because C++ restricts template specialization of member functions)
template<typename Stencil>
inline D_ATTRIBUTE int get_n_stencil( Device_State & state ) // Default
{
    return -1;
};

template<>
inline D_ATTRIBUTE int get_n_stencil<ED_Stencil>( Device_State & state )
{
    return state.n_ed;
};

template<typename Stencil>
inline D_ATTRIBUTE void * __get_stencils( Device_State & state )
{
    return nullptr;
}

template<>
inline D_ATTRIBUTE void * __get_stencils<ED_Stencil>( Device_State & state )
{
    return (void *)state.ed_stencils;
}

template<typename Stencil>
inline D_ATTRIBUTE Stencil * get_stencils( Device_State & state )
{
    return (Stencil *)__get_stencils<Stencil>( state );
}

#endif