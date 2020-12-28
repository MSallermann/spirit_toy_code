#include "implementation/Kernels.hpp"
#ifdef BACKEND_CPU

namespace Spirit
{
namespace Device
{
namespace Kernels
{

void set_gradient_zero( Device::State * state )
{
#pragma omp parallel for
    for( int i = 0; i < state->nos; i++ )
    {
        state->gradient[i] = { 0, 0, 0 };
    }
}

void propagate_spins( Device::State * state )
{
#pragma omp parallel for
    for( int idx = 0; idx < state->nos; idx++ )
    {
        state->spins[idx] += state->timestep * state->gradient[idx];
        state->spins[idx].normalize();
    }
}

} // namespace Kernels
} // namespace Device
} // namespace Spirit
#endif
