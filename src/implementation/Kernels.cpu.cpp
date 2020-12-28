#ifdef BACKEND_CPU
#include "implementation/Kernels.hpp"
namespace Spirit
{
namespace Implementation
{
namespace Kernels
{

void set_gradient_zero( Vector3 * gradient, Interface::State::Geometry & geometry )
{
#pragma omp parallel for
    for( int i = 0; i < geometry.nos; i++ )
    {
        gradient[i] = { 0, 0, 0 };
    }
}

void propagate_spins(
    Vector3 * spins, Vector3 * gradient, Interface::State::Solver_Parameters & solver_parameters, Interface::State::Geometry & geometry )
{
#pragma omp parallel for
    for( int idx = 0; idx < geometry.nos; idx++ )
    {
        spins[idx] += solver_parameters.timestep * gradient[idx];
        spins[idx].normalize();
    }
}

} // namespace Kernels
} // namespace Implementation
} // namespace Spirit
#endif
