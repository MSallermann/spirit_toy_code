#pragma once
#ifndef DEFINITIONS_HPP
#define DEFINITIONS_HPP
#include "Eigen/Dense"

#ifdef BACKEND_CPU
using scalar = double;
#else // BACKEND_CUDA
using scalar = float;
#endif

#ifdef __CUDACC__
#define HD_ATTRIBUTE __host__ __device__
#define D_ATTRIBUTE __device__
#define H_ATTRIBUTE __host__
#else
#define HD_ATTRIBUTE
#define D_ATTRIBUTE
#define H_ATTRIBUTE
#endif

using Matrix3 = Eigen::Matrix<scalar, 3, 3>;
using Vector3 = Eigen::Matrix<scalar, 3, 1>;

struct Pair_Stencil
{
    Pair_Stencil() {}
    Pair_Stencil( int i, int j, int da, int db, int dc, Matrix3 matrix ) : i( i ), j( j ), da( da ), db( db ), dc( dc ), matrix( matrix ) {}
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