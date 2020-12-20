#pragma once
#ifndef CUDA_HELPER_FUNCTIONS
#include <vector>

namespace Cuda_Backend
{

template<typename T>
void malloc_n( T *& dev_ptr, int N )
{
    cudaMalloc( &dev_ptr, N * sizeof( T ) );
}

template<typename T, typename Vec>
void copy_vector_H2D( T * dest_dev_ptr, Vec & src_host_vec )
{
    cudaMemcpy( dest_dev_ptr, src_host_vec.data(), src_host_vec.size() * sizeof( T ), cudaMemcpyHostToDevice );
}

template<typename T, typename Vec>
void copy_vector_D2H( Vec & dest_host_vec, T * src_dev_ptr )
{
    cudaMemcpy( dest_host_vec.data(), src_dev_ptr, dest_host_vec.size() * sizeof( T ), cudaMemcpyDeviceToHost );
}

// Unrolls n nested square loops
inline __device__ void cu_tupel_from_idx( int & idx, int * tupel, int * maxVal, int n )
{
    int idx_diff = idx;
    int div      = 1;
    for( int i = 0; i < n - 1; i++ )
        div *= maxVal[i];
    for( int i = n - 1; i > 0; i-- )
    {
        tupel[i] = idx_diff / div;
        idx_diff -= tupel[i] * div;
        div /= maxVal[i - 1];
    }
    tupel[0] = idx_diff / div;
}

} // namespace Cuda_Backend

#endif