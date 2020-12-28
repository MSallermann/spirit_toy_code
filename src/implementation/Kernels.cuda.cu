#include "implementation/Kernels.hpp"

#ifdef BACKEND_CUDA

namespace Spirit
{
namespace Device
{
namespace Kernels
{

__global__ void cu_set_gradient_zero( Device::State * state )
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for( int i = index; i < state.nos; i += stride )
    {
        state->gradient[i] = { 0, 0, 0 };
    }
}

void set_gradient_zero( Device::State * state )
{
    int blockSize = 1024;
    int numBlocks = ( state.nos + blockSize - 1 ) / blockSize;
    cu_set_gradient_zero<<<blockSize, numBlocks>>>( state );
}

__global__ void cu_propagate_spins( Device::State * state )
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for( int idx = index; idx < state.nos; idx += stride )
    {
        state.spins[idx] += state.timestep * state.gradient[idx];
        state.spins[idx].normalize();
    }
}

void propagate_spins( Device::State * state )
{
    int blockSize = 1024;
    int numBlocks = ( state.nos + blockSize - 1 ) / blockSize;
    cu_propagate_spins<<<blockSize, numBlocks>>>( state );
}

} // namespace Kernels
} // namespace Device
} // namespace Spirit
#endif
