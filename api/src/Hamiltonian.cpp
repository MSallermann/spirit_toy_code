#include "Hamiltonian.h"
#include "Api_State.hpp"
#include "interface/State.hpp"
#include "interface/Stencil.hpp"

#include <array>

template<typename T, int N>
std::array<T, N> get_arr( T in[N] )
{
    std::array<T, N> out;
    for( int i = 0; i < N; i++ )
    {
        out[i] = in[i];
    }
    return out;
}

void Push_ED_Stencil( State * state, ED_Stencil stencil ) noexcept
{
    using Stencil_t = Spirit::Interface::Stencil<2, Spirit::Matrix3>;
    Spirit::Matrix3 dmi_matrix;
    dmi_matrix << 0, 0, 1, 0, 0, 0, -1, 0, 0;

    Spirit::Matrix3 j_matrix = Spirit::Matrix3::Identity();

    state->core_state.hamiltonian.ed_stencils.push_back( Stencil_t(
        stencil.i, get_arr<int, 1>( stencil.j ), get_arr<int, 1>( stencil.da ), get_arr<int, 1>( stencil.db ), get_arr<int, 1>( stencil.dc ),
        stencil.J * j_matrix + stencil.D * dmi_matrix ) );
}

void Set_ED_Stencils( State * state, ED_Stencil * ed_stencils, int N ) noexcept
{

    state->core_state.hamiltonian.ed_stencils.resize( 0 );
    for( int i = 0; i < N; i++ )
    {
        Push_ED_Stencil( state, ed_stencils[i] );
    }
}

void Push_K_Stencil( State * state, K_Stencil stencil ) noexcept
{
    using Stencil_t = Spirit::Interface::Stencil<1, Spirit::Vector3>;
    state->core_state.hamiltonian.k_stencils.push_back( Stencil_t( stencil.i, {}, {}, {}, {}, { stencil.K[0], stencil.K[1], stencil.K[2] } ) );
}

void Set_K_Stencils( State * state, K_Stencil * k_stencils, int N ) noexcept
{
    state->core_state.hamiltonian.k_stencils.resize( 0 );
    for( int i = 0; i < N; i++ )
        Push_K_Stencil( state, k_stencils[i] );
}

void Push_B_Stencil( State * state, B_Stencil stencil ) noexcept
{
    using Stencil_t = Spirit::Interface::Stencil<1, Spirit::Vector3>;
    state->core_state.hamiltonian.b_stencils.push_back( Stencil_t( stencil.i, {}, {}, {}, {}, { stencil.B[0], stencil.B[1], stencil.B[2] } ) );
}

void Set_B_Stencils( State * state, B_Stencil * b_stencils, int N ) noexcept
{
    state->core_state.hamiltonian.b_stencils.resize( 0 );
    for( int i = 0; i < N; i++ )
        Push_B_Stencil( state, b_stencils[i] );
}