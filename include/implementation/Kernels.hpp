#pragma once
#ifndef IMPLEMENTATION_KERNELS_HPP
#define IMPLEMENTATION_KERNELS_HPP

#include "implementation/State.hpp"

namespace Spirit
{
namespace Device
{
namespace Kernels
{

void set_gradient_zero( Device::State * state );
void propagate_spins( Device::State * state );

} // namespace Kernels
} // namespace Device
} // namespace Spirit

#endif