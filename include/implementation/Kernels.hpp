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

struct Summator
{
    int N;
    size_t temp_storage_bytes; // fields needed for cuda
    void * temp_storage;       // fields needed for cuda
    Summator( int N );
    ~Summator();
    void sum( scalar * result, scalar * scalarfield );
};

void dot( scalar * result, Vector3 * vf1, Vector3 * vf2, int N );

void propagate_spins(
    Vector3 * spins, Vector3 * gradient, Interface::State::Solver_Parameters & solver_parameters, Interface::State::Geometry & geometry );

} // namespace Kernels
} // namespace Implementation
} // namespace Spirit

#endif