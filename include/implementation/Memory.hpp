#pragma once
#ifndef MEMORY_HPP
#define MEMORY_HPP

#include "Definitions.hpp"
#include <vector>

#ifdef BACKEND_CUDA
#include "implementation/backend_cuda/cuda_helper_functions.hpp"
#endif

namespace Spirit
{
namespace Device
{

template<typename T>
class device_vector
{
protected:
    T * m_ptr     = nullptr;
    size_t m_size = 0;

public:
    H_ATTRIBUTE
    device_vector() : m_ptr( nullptr ), m_size( 0 ) {}

    H_ATTRIBUTE device_vector( size_t N );

    H_ATTRIBUTE device_vector( const device_vector<T> & old_vector );

    H_ATTRIBUTE device_vector & operator=( const device_vector<T> & old_vector );

    H_ATTRIBUTE void copy_to( std::vector<T> & host_vector );

    H_ATTRIBUTE void copy_from( const std::vector<T> & host_vector );

    H_ATTRIBUTE void copy_to( T * host_ptr );

    H_ATTRIBUTE void copy_from( T * host_vector );

    D_ATTRIBUTE
    T & operator[]( int N );

    HD_ATTRIBUTE
    size_t size() const;

    H_ATTRIBUTE
    ~device_vector();

    HD_ATTRIBUTE
    T * data()
    {
        return m_ptr;
    }
};

#ifdef BACKEND_CPU
template<typename T>
H_ATTRIBUTE device_vector<T>::device_vector( size_t N )
{
    m_ptr  = new T[N];
    m_size = N;
}

template<typename T>
H_ATTRIBUTE device_vector<T>::device_vector( const device_vector<T> & old_vector )
{
    m_size = old_vector.size();
    m_ptr  = new T[m_size];

#pragma omp parallel for
    for( int i = 0; i < m_size; i++ )
    {
        m_ptr[i] = old_vector.m_ptr[i];
    }
}

template<typename T>
H_ATTRIBUTE device_vector<T> & device_vector<T>::operator=( const device_vector<T> & old_vector )
{
    if( this->m_ptr != nullptr )
        delete[] m_ptr;

    m_size = old_vector.size();
    m_ptr  = new T[m_size];

#pragma omp parallel for
    for( int i = 0; i < m_size; i++ )
    {
        m_ptr[i] = old_vector.m_ptr[i];
    }
    return *this;
}

template<typename T>
H_ATTRIBUTE void device_vector<T>::copy_to( std::vector<T> & host_vector )
{
    if( host_vector.data() == this->m_ptr )
    {
        return;
    }
    if( host_vector.size() == this->size() )
    {
#pragma omp parallel for
        for( int i = 0; i < this->size(); i++ )
        {
            host_vector[i] = ( *this )[i];
        }
    }
    else
    {
        throw;
    }
}

template<typename T>
H_ATTRIBUTE void device_vector<T>::copy_to( T * host_ptr )
{
#pragma omp parallel for
    for( int i = 0; i < this->size(); i++ )
    {
        host_ptr[i] = ( *this )[i];
    }
}

template<typename T>
H_ATTRIBUTE void device_vector<T>::copy_from( const std::vector<T> & host_vector )
{
    if( host_vector.data() == this->m_ptr )
    {
        return;
    }
    if( host_vector.size() == this->size() )
    {
#pragma omp parallel for
        for( int i = 0; i < this->size(); i++ )
        {
            ( *this )[i] = host_vector[i];
        }
    }
    else
    {
        throw;
    }
}

template<typename T>
H_ATTRIBUTE void device_vector<T>::copy_from( T * host_ptr )
{
#pragma omp parallel for
    for( int i = 0; i < this->size(); i++ )
    {
        ( *this )[i] = host_ptr[i];
    }
}

template<typename T>
D_ATTRIBUTE T & device_vector<T>::operator[]( int N )
{
    return m_ptr[N];
}

template<typename T>
HD_ATTRIBUTE size_t device_vector<T>::size() const
{
    return m_size;
}

template<typename T>
H_ATTRIBUTE device_vector<T>::~device_vector()
{
    if( m_ptr != nullptr )
    {
        delete[] m_ptr;
    }
}

#else
template<typename T>
H_ATTRIBUTE device_vector<T>::device_vector( size_t N )
{
    CUDA_HELPER::malloc_n( m_ptr, N );
    m_size = N;
}

template<typename T>
H_ATTRIBUTE device_vector<T>::device_vector( const device_vector<T> & old_vector )
{
    m_size = old_vector.size();
    CUDA_HELPER::malloc_n( m_ptr, m_size );
    cudaMemcpy( m_ptr, old_vector.m_ptr, m_size * sizeof( T ), cudaMemcpyDeviceToDevice );
}

template<typename T>
H_ATTRIBUTE device_vector<T> & device_vector<T>::operator=( const device_vector<T> & old_vector )
{
    if( this->m_ptr != nullptr )
        CUDA_HELPER::free( m_ptr );
    m_size = old_vector.size();
    CUDA_HELPER::malloc_n( m_ptr, m_size );
    cudaMemcpy( m_ptr, old_vector.m_ptr, m_size * sizeof( T ), cudaMemcpyDeviceToDevice );
    return *this;
}

template<typename T>
H_ATTRIBUTE void device_vector<T>::copy_to( std::vector<T> & host_vector )
{
    if( this->size() == host_vector.size() )
        CUDA_HELPER::copy_vector_D2H( host_vector, this->m_ptr );
    else
        throw;
}

template<typename T>
H_ATTRIBUTE void device_vector<T>::copy_to( T * host_ptr )
{
    CUDA_HELPER::copy_n_D2H( host_ptr, this->m_ptr, this->size() );
}

template<typename T>
H_ATTRIBUTE void H_ATTRIBUTE device_vector<T>::copy_from( const std::vector<T> & host_vector )
{
    if( this->size() == host_vector.size() )
        CUDA_HELPER::copy_vector_H2D( this->m_ptr, host_vector );
    else
        throw;
}

template<typename T>
H_ATTRIBUTE void device_vector<T>::copy_from( T * host_ptr )
{
    CUDA_HELPER::copy_n_H2D( this->m_ptr, host_ptr, this->size() );
}

template<typename T>
D_ATTRIBUTE T & device_vector<T>::operator[]( int N )
{
    return m_ptr[N];
}

template<typename T>
HD_ATTRIBUTE size_t device_vector<T>::size() const
{
    return m_size;
}

template<typename T>
H_ATTRIBUTE device_vector<T>::~device_vector()
{
    if( m_ptr != nullptr )
        CUDA_HELPER::free( m_ptr );
}
#endif

} // namespace Device
} // namespace Spirit

#endif