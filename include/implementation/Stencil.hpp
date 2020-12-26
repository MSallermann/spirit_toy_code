#pragma once
#ifndef STENCIL_HPP
#define STENCIL_HPP
#include "definitions.hpp"

namespace Spirit
{

template<int N, typename PARAM>
struct Stencil
{
    static constexpr int N_interaction = N;
    using Vector3NArray                = Vector3[N];
    using t_gradient_func              = Vector3( Vector3[N], PARAM );
    using t_energy_func                = scalar( Vector3[N], PARAM );
    using t_energy_from_gradient_func  = scalar( Vector3[N], PARAM );

    int i;
    std::array<int, N - 1> j;
    std::array<int, N - 1> da;
    std::array<int, N - 1> db;
    std::array<int, N - 1> dc;

    PARAM param;

    Stencil(){};
    Stencil( int i, PARAM param ) : i( i ), param( param ){};
    Stencil( int i, std::array<int, N - 1> j, std::array<int, N - 1> da, std::array<int, N - 1> db, std::array<int, N - 1> dc, PARAM param )
            : i( i ), j( j ), da( da ), db( db ), dc( dc ), param( param ){};

    HD_ATTRIBUTE
    Vector3 gradient( const Vector3NArray & interaction_spins );

    HD_ATTRIBUTE
    scalar energy( const Vector3NArray & interaction_spins );

    HD_ATTRIBUTE
    scalar energy_from_gradient( const Vector3 & gradient, const Vector3NArray & interaction_spins )
    {
        return energy( interaction_spins );
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