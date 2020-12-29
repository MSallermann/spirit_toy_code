#ifdef BACKEND_CUDA

#include "implementation/Kernels.hpp"
#include "implementation/Lambda.hpp"

#include <cub/cub.cuh>

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

Summator::Summator( int N ) : N( N )
{
    temp_storage       = NULL;
    temp_storage_bytes = 0;
    cub::DeviceReduce::Sum( temp_storage, temp_storage_bytes, (scalar *)nullptr, (scalar *)nullptr, N );
    cudaMalloc( &temp_storage, temp_storage_bytes );
};

Summator::~Summator()
{
    cudaFree( this->temp_storage );
};

void Summator::sum( scalar * result, scalar * scalarfield )
{
    cub::DeviceReduce::Sum( temp_storage, temp_storage_bytes, scalarfield, result, N );
    last_result = result;
};

scalar Summator::download_last_result()
{
    scalar temp;
    CUDA_HELPER::copy_D2H( &temp, last_result );
    return temp;
}

void dot( scalar * result, Vector3 * vf1, Vector3 * vf2, int N )
{
    Lambda::apply( N, [result, vf1, vf2] SPIRIT_LAMBDA( int idx ) { result[idx] = vf1[idx].dot( vf2[idx] ); } );
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
