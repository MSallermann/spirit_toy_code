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

void Set_ED_Stencils( State * state, ED_Stencil * ed_stencils, int N ) noexcept
{
    using Stencil_t = Spirit::Interface::Stencil<2, Spirit::Matrix3>;
    Spirit::Matrix3 dmi_matrix;
    dmi_matrix << 0, 0, 1, 0, 0, 0, -1, 0, 0;

    Spirit::Matrix3 j_matrix = Spirit::Matrix3::Identity();

    std::vector<Stencil_t> stencil_vec;
    for( int i = 0; i < N; i++ )
    {
        auto & stencil = ed_stencils[i];
        stencil_vec.push_back( Stencil_t(
            stencil.i, get_arr<int, 1>( stencil.j ), get_arr<int, 1>( stencil.da ), get_arr<int, 1>( stencil.db ), get_arr<int, 1>( stencil.dc ),
            stencil.J * j_matrix + stencil.D * dmi_matrix ) );
    }
    state->core_state.hamiltonian.ed_stencils = stencil_vec;
}

void Set_K_Stencils( State * state, K_Stencil * k_stencils, int N ) noexcept
{
    using Stencil_t = Spirit::Interface::Stencil<1, Spirit::Vector3>;
    std::vector<Stencil_t> stencil_vec;
    for( int i = 0; i < N; i++ )
    {
        auto & stencil = k_stencils[i];
        stencil_vec.push_back( Stencil_t( stencil.i, {}, {}, {}, {}, { stencil.K[0], stencil.K[1], stencil.K[2] } ) );
    }
    state->core_state.hamiltonian.k_stencils = stencil_vec;
}

void Set_B_Stencils( State * state, B_Stencil * b_stencils, int N ) noexcept
{
    using Stencil_t = Spirit::Interface::Stencil<1, Spirit::Vector3>;
    std::vector<Stencil_t> stencil_vec;
    for( int i = 0; i < N; i++ )
    {
        auto & stencil = b_stencils[i];
        stencil_vec.push_back( Stencil_t( stencil.i, {}, {}, {}, {}, { stencil.B[0], stencil.B[1], stencil.B[2] } ) );
    }
    state->core_state.hamiltonian.b_stencils = stencil_vec;
}