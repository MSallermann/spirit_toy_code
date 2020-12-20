#ifdef BACKEND_CUDA
#include "State.hpp"
#include "backend_cuda/cuda_helper_functions.hpp"
#include "backend_handle.hpp"

Backend_Handle::Backend_Handle( State & state )
{
    this->n_cell_atoms = state.n_cell_atoms;
    this->N_pair       = state.pair_stencils.size();
    this->timestep     = state.timestep;
    this->nos          = state.Nos();

    // Allocate device memory
    Cuda_Backend::malloc_n( this->n_cells, state.n_cells.size() );
    Cuda_Backend::malloc_n( this->spins, state.spins.size() );
    Cuda_Backend::malloc_n( this->pair_stencils, state.pair_stencils.size() );

    // Copy to device
    Cuda_Backend::copy_vector_H2D( this->n_cells, state.n_cells );
    Cuda_Backend::copy_vector_H2D( this->spins, state.spins );
    Cuda_Backend::copy_vector_H2D( this->pair_stencils, state.pair_stencils );

    // Gradient is only computed on device, therefore no copy
    Cuda_Backend::malloc_n( this->gradient, this->nos );

    // Create the device ptr for this backend handle (copies all the pointer adresses etc. from host to device)
    cudaMalloc( &this->dev_ptr, sizeof( Backend_Handle ) );
    cudaMemcpy( this->dev_ptr, this, sizeof( Backend_Handle ), cudaMemcpyHostToDevice );
}

void Backend_Handle::Upload( State & state )
{
    Cuda_Backend::copy_vector_H2D( this->spins, state.spins );
}

void Backend_Handle::Download( State & state )
{
    Cuda_Backend::copy_vector_D2H( state.spins, this->spins );
}

Backend_Handle::~Backend_Handle()
{
    cudaFree( n_cells );
    cudaFree( spins );
    cudaFree( pair_stencils );
    cudaFree( gradient );
    cudaFree( dev_ptr );
}

#endif