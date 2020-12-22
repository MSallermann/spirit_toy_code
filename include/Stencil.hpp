#pragma once
#ifndef STENCIL_HPP
#define STENCIL_HPP
#include "definitions.hpp"

namespace Spirit
{

template<int N, typename PARAM>
struct Stencil
{
    using Vector3NArray               = Vector3[N];
    using t_gradient_func             = Vector3( Vector3, Vector3[N], PARAM );
    using t_energy_func               = scalar( Vector3, Vector3[N], PARAM );
    using t_energy_from_gradient_func = scalar( Vector3, Vector3, Vector3[N], PARAM );

    int i;
    std::array<int, N> j;
    std::array<int, N> da;
    std::array<int, N> db;
    std::array<int, N> dc;

    PARAM param;

    Stencil(){};
    Stencil( int i, PARAM param ) : i( i ), param( param ){};
    Stencil( int i, std::array<int, N> j, std::array<int, N> da, std::array<int, N> db, std::array<int, N> dc, PARAM param ) : i( i ), j( j ), da( da ), db( db ), dc( dc ), param( param ){};

    HD_ATTRIBUTE
    Vector3 gradient( const Vector3 & spin, const Vector3NArray & interaction_spins );

    HD_ATTRIBUTE
    scalar energy( const Vector3 & spin, const Vector3NArray & interaction_spins );

    HD_ATTRIBUTE
    scalar energy_from_gradient( const Vector3 & spin, const Vector3 & gradient, const Vector3NArray & interaction_spins )
    {
        return energy( spin, interaction_spins );
    };

    HD_ATTRIBUTE
    int get_i()
    {
        return i;
    };

    HD_ATTRIBUTE
    int get_j( int idx )
    {
        return ( (int *)&( this->j ) )[idx];
    };

    HD_ATTRIBUTE
    int get_da( int idx )
    {
        return ( (int *)&( this->da ) )[idx];
    };

    HD_ATTRIBUTE
    int get_db( int idx )
    {
        return ( (int *)&( this->db ) )[idx];
    };

    HD_ATTRIBUTE
    int get_dc( int idx )
    {
        return ( (int *)&( this->db ) )[idx];
    };
};

} // namespace Spirit
#endif