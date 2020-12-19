#ifdef BACKEND_CUDA
#include "State.hpp"
#include "backend_cuda/cuda_helper_functions.hpp"
#include "backend_handle.hpp"

Backend_Handle::Backend_Handle() {}

void Backend_Handle::Allocate( State & state )
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
    if( this->gradient != nullptr )
        cudaFree( this->gradient );
    Cuda_Backend::malloc_n( &this->gradient, this->nos );
}

void Backend_Handle::Upload( State & state )
{
    // Nothing to be done
}

void Backend_Handle::Download( State & state )
{
    // Nothing to be done
}

Backend_Handle::~Backend_Handle()
{
    cudaFree( n_cells );
    cudaFree( spins );
    cudaFree( pair_stencils );
    cudaFree( gradient );
}

#endif