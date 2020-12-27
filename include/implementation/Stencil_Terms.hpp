#pragma once
#ifndef IMPLEMENTATION_STENCIL_TERMS
#define IMPLEMENTATION_STENCIL_TERMS
#include "implementation/Stencil.hpp"
#include <array>

namespace Spirit
{
namespace Device
{
struct ED_Stencil : Stencil<2, Matrix3>
{
    ED_Stencil() = default;

    ED_Stencil( int i, std::array<int, 1> j, std::array<int, 1> da, std::array<int, 1> db, std::array<int, 1> dc, Matrix3 param )
            : Stencil<2, Matrix3>( i, j, da, db, dc, param ){};

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

struct Bfield_Stencil : Stencil<1, Vector3>
{
    Bfield_Stencil() = default;

    Bfield_Stencil( int i, std::array<int, 0> j, std::array<int, 0> da, std::array<int, 0> db, std::array<int, 0> dc, Vector3 param )
            : Stencil<1, Vector3>( i, j, da, db, dc, param ){};

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

struct K_Stencil : Stencil<1, Vector3>
{
    K_Stencil() = default;
    K_Stencil( int i, std::array<int, 0> j, std::array<int, 0> da, std::array<int, 0> db, std::array<int, 0> dc, Vector3 param )
            : Stencil<1, Vector3>( i, j, da, db, dc, param ){};

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