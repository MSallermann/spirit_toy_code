#pragma once
#ifndef IMPLEMENTATION_STENCIL_TERMS
#define IMPLEMENTATION_STENCIL_TERMS
#include "implementation/Stencil.hpp"
#include <array>

namespace Spirit
{
namespace Device
{

template<int N, typename PARAM>
struct StencilImp : Stencil<N, PARAM>
{
    using Vector3NArray = Vector3[N];
    StencilImp()        = default;
    StencilImp( int i, std::array<int, N - 1> j, std::array<int, N - 1> da, std::array<int, N - 1> db, std::array<int, N - 1> dc, PARAM param )
            : Stencil<N, PARAM>( i, j, da, db, dc, param ){};

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

    HD_ATTRIBUTE
    virtual Vector3 gradient( const Vector3NArray & interaction_spins ) = 0;

    HD_ATTRIBUTE
    virtual scalar energy( const Vector3NArray & interaction_spins ) = 0;
};

struct ED_Stencil : StencilImp<2, Matrix3>
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

struct Bfield_Stencil : StencilImp<1, Vector3>
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

struct K_Stencil : StencilImp<1, Vector3>
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

} // namespace Device
} // namespace Spirit

#endif