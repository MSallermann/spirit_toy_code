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

#endif