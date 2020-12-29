#pragma once
#ifndef IMPLEMENTATION_LAMBDA_HPP

#include "Definitions.hpp"
#include "implementation/Memory.hpp"

#ifdef BACKEND_CUDA
#include "implementation/backend_cuda/cuda_helper_functions.hpp"
// #include <cub/cub.cuh>
#define SPIRIT_LAMBDA __device__
#else
#define SPIRIT_LAMBDA
#endif

namespace Spirit
{
namespace Implementation
{
namespace Lambda
{

#ifdef BACKEND_CUDA

template<typename F>
__global__ void cu_apply( int N, F f )
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if( idx < N )
        f( idx );
}

template<typename F>
void apply( int N, F f )
{
    cu_apply<<<( N + 1023 ) / 1024, 1024>>>( N, f );
}

#else

template<typename F>
void apply( int N, const F & f )
{
#pragma omp parallel for
    for( int idx = 0; idx < N; ++idx )
    {
        f( idx );
    }
}

#endif
} // namespace Lambda
} // namespace Implementation
} // namespace Spirit

#endif
