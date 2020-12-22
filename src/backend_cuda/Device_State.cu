#ifdef BACKEND_CUDA

#include "Device_State.hpp"
#include "Host_State.hpp"
#include "backend_cuda/cuda_helper_functions.hpp"

void Device_State::allocate( Host_State * host_state )
{
    this->free();
    allocated = true;

    this->allocated     = true;
    this->nos           = host_state->nos;
    this->n_cells_total = host_state->n_cells_total;
    this->n_cell_atoms  = host_state->n_cell_atoms;
    this->n_cells[0]    = host_state->n_cells[0];
    this->n_cells[1]    = host_state->n_cells[1];
    this->n_cells[2]    = host_state->n_cells[2];
    this->n_ed          = host_state->ed_stencils.size();
    this->timestep      = host_state->timestep;
    CUDA_HELPER::malloc_n( this->gradient, host_state->nos );
    CUDA_HELPER::malloc_n( this->spins, host_state->nos );
    CUDA_HELPER::malloc_n( this->ed_stencils, host_state->ed_stencils.size() );
}

void Device_State::download( Host_State * host_state )
{
    CUDA_HELPER::copy_vector_D2H( host_state->spins, this->spins );
    CUDA_HELPER::copy_vector_D2H( host_state->gradient, this->gradient );
}

void Device_State::upload( Host_State * host_state )
{
    CUDA_HELPER::copy_vector_H2D( this->spins, host_state->spins );
    CUDA_HELPER::copy_vector_H2D( this->ed_stencils, host_state->ed_stencils );
}

void Device_State::free()
{
    if( !allocated )
        return;
    allocated = false;
    printf( "Freeing backend resources\n" );
    CUDA_HELPER::free( spins );
    CUDA_HELPER::free( gradient );
    CUDA_HELPER::free( ed_stencils );
}

#endif