#pragma once
#ifndef IMPLEMENTATION_STENCIL_TERMS
#define IMPLEMENTATION_STENCIL_TERMS
#include "interface/Stencil.hpp"
#include <array>

namespace Spirit
{
namespace Implementation
{

template<int N, typename PARAM>
struct StencilImp : public Interface::Stencil<N, PARAM>
{
    using Vector3NArray = Vector3[N];
    StencilImp()        = default;
    StencilImp( int i, std::array<int, N - 1> j, std::array<int, N - 1> da, std::array<int, N - 1> db, std::array<int, N - 1> dc, PARAM param )
            : Interface::Stencil<N, PARAM>( i, j, da, db, dc, param ){};

    H_ATTRIBUTE void read_from_host( const Spirit::Interface::Stencil<N, PARAM> & interface_stencil )
    {
        this->i     = interface_stencil.i;
        this->j     = interface_stencil.j;
        this->da    = interface_stencil.da;
        this->db    = interface_stencil.db;
        this->dc    = interface_stencil.dc;
        this->param = interface_stencil.param;
    }

    D_ATTRIBUTE
    inline int get_i()
    {
        return this->i;
    }

    D_ATTRIBUTE
    inline int get_j( int idx )
    {
        return ( (int *)&this->j )[idx];
    }

    D_ATTRIBUTE
    inline int get_da( int idx )
    {
        return ( (int *)&this->da )[idx];
    }

    D_ATTRIBUTE
    inline int get_db( int idx )
    {
        return ( (int *)&this->db )[idx];
    }

    D_ATTRIBUTE
    inline int get_dc( int idx )
    {
        return ( (int *)&this->dc )[idx];
    }
};

struct ED_Stencil : public StencilImp<2, Matrix3>
{
    ED_Stencil() = default;

    ED_Stencil( int i, std::array<int, 1> j, std::array<int, 1> da, std::array<int, 1> db, std::array<int, 1> dc, Matrix3 param )
            : StencilImp<2, Matrix3>( i, j, da, db, dc, param ){};

    HD_ATTRIBUTE
    Vector3 gradient( const Vector3NArray & interaction_spins )
    {
        return param * interaction_spins[1];
    }

    HD_ATTRIBUTE
    scalar energy( const Vector3NArray & interaction_spins )
    {
        return interaction_spins[0].transpose() * param * interaction_spins[1];
    }
};

struct Bfield_Stencil : public StencilImp<1, Vector3>
{
    Bfield_Stencil() = default;

    Bfield_Stencil( int i, std::array<int, 0> j, std::array<int, 0> da, std::array<int, 0> db, std::array<int, 0> dc, Vector3 param )
            : StencilImp<1, Vector3>( i, j, da, db, dc, param ){};

    HD_ATTRIBUTE
    Vector3 gradient( const Vector3NArray & interaction_spins )
    {
        return param;
    }

    HD_ATTRIBUTE
    scalar energy( const Vector3NArray & interaction_spins )
    {
        return param.dot( interaction_spins[0] );
    }
};

struct K_Stencil : public StencilImp<1, Vector3>
{
    K_Stencil() = default;
    K_Stencil( int i, std::array<int, 0> j, std::array<int, 0> da, std::array<int, 0> db, std::array<int, 0> dc, Vector3 param )
            : StencilImp<1, Vector3>( i, j, da, db, dc, param ){};

    HD_ATTRIBUTE
    Vector3 gradient( const Vector3NArray & interaction_spins )
    {
        return 2 * param * ( param.normalized().dot( interaction_spins[0] ) );
    }

    HD_ATTRIBUTE
    scalar energy( const Vector3NArray & interaction_spins )
    {
        return param.norm() * ( param.normalized().dot( interaction_spins[0] ) ) * ( param.normalized().dot( interaction_spins[0] ) );
    }
};

} // namespace Implementation
} // namespace Spirit

#endif