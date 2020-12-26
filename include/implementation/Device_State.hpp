#pragma once
#ifndef DEVICE_STATE_HPP
#define DEVICE_STATE_HPP
#include "definitions.hpp"
#include "implementation/Hamiltonian.hpp"

namespace Spirit
{

namespace Host
{
class Host_State;
}

namespace Device
{

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

    int n_b;
    Bfield_Stencil * b_stencils;

    int n_k;
    K_Stencil * k_stencils;

    void H_ATTRIBUTE allocate( Spirit::Host::Host_State * host_state );
    void H_ATTRIBUTE upload( Spirit::Host::Host_State * host_state );
    void H_ATTRIBUTE download( Spirit::Host::Host_State * host_state );
    void H_ATTRIBUTE free();
};

// Functions to read Stencils from Device state (They are not defined as class members because C++ restricts template specialization of member functions)
template<typename Stencil>
inline D_ATTRIBUTE int get_n_stencil( Device_State & state ) // Default
{
    return -1;
};

template<typename Stencil>
inline D_ATTRIBUTE void * __get_stencils( Device_State & state )
{
    return nullptr;
}

template<>
inline D_ATTRIBUTE int get_n_stencil<ED_Stencil>( Device_State & state )
{
    return state.n_ed;
};

template<>
inline D_ATTRIBUTE void * __get_stencils<ED_Stencil>( Device_State & state )
{
    return (void *)state.ed_stencils;
}

template<>
inline D_ATTRIBUTE int get_n_stencil<K_Stencil>( Device_State & state )
{
    return state.n_k;
};

template<>
inline D_ATTRIBUTE void * __get_stencils<K_Stencil>( Device_State & state )
{
    return (void *)state.k_stencils;
}

template<>
inline D_ATTRIBUTE int get_n_stencil<Bfield_Stencil>( Device_State & state )
{
    return state.n_b;
};

template<>
inline D_ATTRIBUTE void * __get_stencils<Bfield_Stencil>( Device_State & state )
{
    return (void *)state.b_stencils;
}

template<typename Stencil>
inline D_ATTRIBUTE Stencil * get_stencils( Device_State & state )
{
    return (Stencil *)__get_stencils<Stencil>( state );
}

} // namespace Device
} // namespace Spirit
#endif