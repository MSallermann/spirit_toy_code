#ifdef BACKEND_CUDA

#include "implementation/Kernels.hpp"

namespace Spirit
{
namespace Implementation
{
namespace Kernels
{

__global__ void cu_set_gradient_zero( Vector3 * gradient, State_Pod state )
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for( int i = index; i < state.nos; i += stride )
    {
        gradient[i] = { 0, 0, 0 };
    }
}

void set_gradient_zero( Vector3 * gradient, State_Pod & state )
{
    int blockSize = 1024;
    int numBlocks = ( state.nos + blockSize - 1 ) / blockSize;
    cu_set_gradient_zero<<<blockSize, numBlocks>>>( gradient, state );
}

__global__ void cu_propagate_spins( Vector3 * spins, Vector3 * gradient, State_Pod state )
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for( int idx = index; idx < state.nos; idx += stride )
    {
        spins[idx] += state.timestep * gradient[idx];
        spins[idx].normalize();
    }
}

void propagate_spins( Vector3 * spins, Vector3 * gradient, State_Pod & state )
{
    int blockSize = 1024;
    int numBlocks = ( state.nos + blockSize - 1 ) / blockSize;
    cu_propagate_spins<<<blockSize, numBlocks>>>( spins, gradient, state );
}

} // namespace Kernels
} // namespace Implementation
} // namespace Spirit
#endif
