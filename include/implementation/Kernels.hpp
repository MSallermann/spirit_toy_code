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
void propagate_spins(
    Vector3 * spins, Vector3 * gradient, Interface::State::Solver_Parameters & solver_parameters, Interface::State::Geometry & geometry );

} // namespace Kernels
} // namespace Implementation
} // namespace Spirit

#endif