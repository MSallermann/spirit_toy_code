#pragma once
#ifndef INTERFACE_STENCIL_HPP
#define INTERFACE_STENCIL_HPP
#include "Definitions.hpp"
#include <array>

namespace Spirit
{
namespace Interface
{

template<int N, typename PARAM>
struct Stencil
{
public:
    static constexpr int N_interaction = N;
    int i;
    std::array<int, N - 1> j;
    std::array<int, N - 1> da;
    std::array<int, N - 1> db;
    std::array<int, N - 1> dc;

    PARAM param;

    Stencil(){};
    Stencil( int i, std::array<int, N - 1> j, std::array<int, N - 1> da, std::array<int, N - 1> db, std::array<int, N - 1> dc, PARAM param )
            : i( i ), j( j ), da( da ), db( db ), dc( dc ), param( param ){};
};

} // namespace Interface
} // namespace Spirit
#endif