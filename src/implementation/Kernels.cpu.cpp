#ifdef BACKEND_CPU
#include "implementation/Kernels.hpp"
namespace Spirit
{
namespace Device
{
namespace Kernels
{

void set_gradient_zero( Vector3 * gradient, State_Pod & state )
{
#pragma omp parallel for
    for( int i = 0; i < state.nos; i++ )
    {
        gradient[i] = { 0, 0, 0 };
    }
}

void propagate_spins( Vector3 * spins, Vector3 * gradient, State_Pod & state )
{
#pragma omp parallel for
    for( int idx = 0; idx < state.nos; idx++ )
    {
        spins[idx] += state.timestep * gradient[idx];
        spins[idx].normalize();
    }
}

} // namespace Kernels
} // namespace Device
} // namespace Spirit
#endif
