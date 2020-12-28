#pragma once
#ifndef IMPLEMENTATION_KERNELS_HPP
#define IMPLEMENTATION_KERNELS_HPP

#include "implementation/State.hpp"

namespace Spirit
{
namespace Implementation
{
namespace Kernels
{

void set_gradient_zero( Vector3 * gradient, State_Pod & state );
void propagate_spins( Vector3 * spins, Vector3 * gradient, State_Pod & state );

} // namespace Kernels
} // namespace Implementation
} // namespace Spirit

#endif