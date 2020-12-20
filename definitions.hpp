#pragma once
#ifndef DEFINITIONS_HPP
#define DEFINITIONS_HPP
#include "Eigen/Dense"

#ifdef BACKEND_CPU
using scalar = double;
#else
using scalar = float;
#endif

using Matrix3 = Eigen::Matrix<scalar, 3, 3>;
using Vector3 = Eigen::Matrix<scalar, 3, 1>;

struct Pair_Stencil
{
    int i;
    int j;
    int da;
    int db;
    int dc;
    Matrix3 matrix;
};

#ifdef DEBUG_PRINT
#include <iostream>
#include <string>
#endif
template<typename T>
inline void debug_print( T msg )
{
#ifdef DEBUG_PRINT
    std::cout << msg << "\n";
#endif
}

#endif