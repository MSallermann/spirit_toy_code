#ifdef BACKEND_CUDA
#include "backend.hpp"
#include "backend_cuda/cuda_helper_functions.hpp"
#include <iostream>

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

    int Na            = state->n_cells[0];
    int Nb            = state->n_cells[1];
    int Nc            = state->n_cells[2];
    int N_cells_total = Na * Nb * Nc;

    for( int i = index; i < state->nos; i += stride )
    {
        state->gradient[i] = { 0, 0, 0 };
    }

    for( int i_cell = index; i_cell < N_cells_total; i_cell += stride )
    {
        int tupel[3];
        Cuda_Backend::cu_tupel_from_idx( i_cell, tupel, state->n_cells, 3 ); // tupel now is {i, a, b, c}
        int a = tupel[0];
        int b = tupel[1];
        int c = tupel[2];
        for( int i = 0; i < state->n_cell_atoms; i++ )
        {
            int idx_i = i + state->n_cell_atoms * ( i_cell );
            for( int p = 0; p < state->N_pair; p++ )
            {
                const Pair_Stencil & pair = state->pair_stencils[p];
                int idx_j                 = pair.j + state->n_cell_atoms * ( ( a + pair.da ) + Na * ( b + pair.db + Nc * ( c + pair.dc ) ) );
                if( i == pair.i && idx_j > 0 && idx_j < state->nos )
                {
                    state->gradient[idx_i] += pair.matrix * state->spins[idx_j];
                }
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
        if( iter % state.n_log == 0 )
        {
            printf( "iter = %i\n", iter );
            state.backend->Download( state );
            std::cout << "Spin[0,0,0] = " << state.spins[0].transpose() << "\n";
        }
    }
    cudaDeviceSynchronize();
    state.backend->Download( state );
}

#endif