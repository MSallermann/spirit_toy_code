#pragma once
#ifndef IMPLEMENTATION_KERNELS_HPP
#define IMPLEMENTATION_KERNELS_HPP

#include "interface/State.hpp"

namespace Spirit
{
namespace Implementation
{
namespace Kernels
{

void set_gradient_zero( Vector3 * gradient, Interface::State::Geometry & geometry );

struct Sum_Configuration
{
    size_t temp_storage_bytes;
    void * temp_storage;
    static Sum_Configuration none()
    {
        return Sum_Configuration{ 0, nullptr };
    }
    ~Sum_Configuration();
};

struct Summator
{
    int N;
    size_t temp_storage_bytes; // fields needed for cuda
    void * temp_storage;       // fields needed for cuda
    scalar * last_result;      // device pointer to last computed result
    Summator( int N );
    ~Summator();
    void sum( scalar * result, scalar * scalarfield );
    scalar download_last_result(); // downloads last result to host and returns it
};

void dot( scalar * result, Vector3 * vf1, Vector3 * vf2, int N );

void propagate_spins(
    Vector3 * spins, Vector3 * gradient, Interface::State::Solver_Parameters & solver_parameters, Interface::State::Geometry & geometry );

} // namespace Kernels
} // namespace Implementation
} // namespace Spirit

#endif