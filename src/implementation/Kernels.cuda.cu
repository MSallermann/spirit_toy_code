#ifdef BACKEND_CUDA

#include "implementation/Kernels.hpp"

namespace Spirit
{
namespace Implementation
{
namespace Kernels
{

__global__ void cu_set_gradient_zero( Vector3 * gradient, Interface::State::Geometry geometry )
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for( int i = index; i < geometry.nos; i += stride )
    {
        gradient[i] = { 0, 0, 0 };
    }
}

void set_gradient_zero( Vector3 * gradient, Interface::State::Geometry & geometry )
{
    int blockSize = 1024;
    int numBlocks = ( geometry.nos + blockSize - 1 ) / blockSize;
    cu_set_gradient_zero<<<blockSize, numBlocks>>>( gradient, geometry );
}

__global__ void
cu_propagate_spins( Vector3 * spins, Vector3 * gradient, Interface::State::Solver_Parameters solver_parameters, Interface::State::Geometry geometry )
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for( int idx = index; idx < geometry.nos; idx += stride )
    {
        spins[idx] += solver_parameters.timestep * gradient[idx];
        spins[idx].normalize();
    }
}

void propagate_spins(
    Vector3 * spins, Vector3 * gradient, Interface::State::Solver_Parameters & solver_parameters, Interface::State::Geometry & geometry )
{
    int blockSize = 1024;
    int numBlocks = ( geometry.nos + blockSize - 1 ) / blockSize;
    cu_propagate_spins<<<blockSize, numBlocks>>>( spins, gradient, solver_parameters, geometry );
}

} // namespace Kernels
} // namespace Implementation
} // namespace Spirit
#endif
