#ifdef BACKEND_CUDA
#include "backend.hpp"
#include "backend_cuda/cuda_helper_functions.hpp"

std::string description()
{
    std::string des;
    des = "CUDA";
    return des;
}

__global__ void gradient( Backend_Handle * state )
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int Na = state->n_cells[0], Nb = state->n_cells[1]; //, Nc = state.n_cells[2];
    int i, a, b, c;
    int idx_j;

    for( int i = index; i < state->nos; i += stride )
    {
        state->gradient[i] = { 0, 0, 0 };
    }

    for( int idx_i = index; idx_i < state->nos; idx_i += stride )
    {
        int tupel[4];
        Cuda_Backend::cu_tupel_from_idx( idx_i, tupel, state->n_cells, 4 ); // tupel now is {i, a, b, c}
        i = tupel[0];
        a = tupel[1];
        b = tupel[2];
        c = tupel[3];
        for( int p = 0; p < state->N_pair; p++ )
        {
            const Pair_Stencil & pair = state->pair_stencils[p];
            idx_j                     = pair.j + state->n_cell_atoms * ( ( a + pair.da ) + Na * ( b + pair.db + Nb * ( c + pair.dc ) ) );

            if( i == pair.i && idx_j > 0 && idx_j < state->nos )
            {
                state->gradient[idx_i] += pair.matrix * state->spins[idx_j];
            }
        }
    }
}

__global__ void propagate_spins( Backend_Handle * state )
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for( int idx = index; idx < state->nos; idx += stride )
    {
        state->spins[idx] += state->timestep * state->gradient[idx];
        state->spins[idx].normalize();
    }
}

void iterate( State & state, int N_iterations )
{

    int blockSize = 1024;
    int numBlocks = ( state.Nos() + blockSize - 1 ) / blockSize;

    for( int iter = 0; iter < N_iterations; iter++ )
    {
        gradient<<<numBlocks, blockSize>>>( state.backend->dev_ptr );
        propagate_spins<<<numBlocks, blockSize>>>( state.backend->dev_ptr );
    }
    cudaDeviceSynchronize();
    state.backend->Download( state );
}

#endif