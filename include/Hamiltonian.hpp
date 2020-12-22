#pragma once
#ifndef HAMILTONIAN_HPP
#define HAMILTONIAN_HPP
#include "Stencil.hpp"

struct ED_Stencil : Stencil<1, Matrix3>
{
    ED_Stencil( int i, std::array<int, 1> j, std::array<int, 1> da, std::array<int, 1> db, std::array<int, 1> dc, Matrix3 param ) : Stencil<1, Matrix3>( i, j, da, db, dc, param ){};

    HD_ATTRIBUTE
    Vector3 gradient( const Vector3 & spin, const Vector3NArray & interaction_spins )
    {
        return param * interaction_spins[0];
    }

    HD_ATTRIBUTE
    scalar energy( const Vector3 & spin, const Vector3NArray & interaction_spins )
    {
        return spin.transpose() * param * interaction_spins[0];
    }
};

struct Bfield_Stencil : Stencil<0, Vector3>
{
    Bfield_Stencil( int i, std::array<int, 0> j, std::array<int, 0> da, std::array<int, 0> db, std::array<int, 0> dc, Vector3 param ) : Stencil<0, Vector3>( i, j, da, db, dc, param ){};

    HD_ATTRIBUTE
    Vector3 gradient( const Vector3 & spin, const Vector3NArray & interaction_spins )
    {
        return param;
    }

    HD_ATTRIBUTE
    scalar energy( const Vector3 & spin, const Vector3NArray & interaction_spins )
    {
        return param.dot( spin );
    }
};

struct K_Stencil : Stencil<0, Vector3>
{
    K_Stencil( int i, std::array<int, 0> j, std::array<int, 0> da, std::array<int, 0> db, std::array<int, 0> dc, Vector3 param ) : Stencil<0, Vector3>( i, j, da, db, dc, param ){};

    HD_ATTRIBUTE
    Vector3 gradient( const Vector3 & spin, const Vector3NArray & interaction_spins )
    {
        return 2 * param * ( param.normalized().dot( spin ) );
    }

    HD_ATTRIBUTE
    scalar energy( const Vector3 & spin, const Vector3NArray & interaction_spins )
    {
        return param.norm() * ( param.normalized().dot( spin ) ) * ( param.normalized().dot( spin ) );
    }
};

#endif