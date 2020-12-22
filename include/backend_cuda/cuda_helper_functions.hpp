#pragma once
#ifndef CUDA_HELPER_FUNCTIONS
#include <vector>

#define gpuErrchk( ans )                                                                                                                                                                               \
    {                                                                                                                                                                                                  \
        gpuAssert( ( ans ), __FILE__, __LINE__, __FUNCTION__ );                                                                                                                                        \
    }

inline void gpuAssert( cudaError_t code, const char * file, int line, const char * function, bool abort = true )
{
    if( code != cudaSuccess )
    {
        fprintf( stderr, "GPU ASSERT ERROR: %s in function '%s' in %s(%d)\n", cudaGetErrorString( code ), function, file, line );
        if( abort )
            exit( code );
    }
}

namespace CUDA_HELPER
{

// Some helper functions
template<typename T>
void malloc_n( T *& dev_ptr, size_t N )
{
    auto err = cudaMalloc( &dev_ptr, N * sizeof( T ) );
    gpuErrchk( err );
}

template<typename T, typename Vec>
void copy_vector_H2D( T * dest_dev_ptr, Vec & src_host_vec )
{
    auto err = cudaMemcpy( dest_dev_ptr, src_host_vec.data(), src_host_vec.size() * sizeof( T ), cudaMemcpyHostToDevice );
    gpuErrchk( err );
}

template<typename T, typename Vec>
void copy_vector_D2H( Vec & dest_host_vec, T * src_dev_ptr )
{
    auto err = cudaMemcpy( dest_host_vec.data(), src_dev_ptr, dest_host_vec.size() * sizeof( T ), cudaMemcpyDeviceToHost );
    gpuErrchk( err );
}

template<typename T>
void copy_D2H( T * dest_host_ptr, T * src_dev_ptr )
{
    auto err = cudaMemcpy( dest_host_ptr, src_dev_ptr, sizeof( T ), cudaMemcpyDeviceToHost );
    gpuErrchk( err );
}

template<typename T>
void copy_H2D( T * dest_dev_ptr, T * src_host_ptr )
{
    auto err = cudaMemcpy( dest_dev_ptr, src_host_ptr, sizeof( T ), cudaMemcpyHostToDevice );
    gpuErrchk( err );
}

template<typename T>
void free( T dev_ptr )
{
    cudaFree( dev_ptr );
}

// Unrolls n nested, square loops
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

} // namespace CUDA_HELPER
#endif