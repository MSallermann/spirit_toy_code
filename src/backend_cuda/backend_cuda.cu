#ifdef BACKEND_CUDA
#include "backend.hpp"
#include "backend_cuda/cuda_helper_functions.hpp"
#include <iostream>

namespace Spirit
{
namespace Device
{

std::string description()
{
    std::string des;
    des = "CUDA";
    return des;
}

__global__ void set_gradient_zero( Device_State state )
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for( int i = index; i < state.nos; i += stride )
    {
        state.gradient[i] = { 0, 0, 0 };
    }
}

template<int N, typename Stencil>
__global__ void stencil_gradient( Device_State state )
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    auto N_Stencil = get_n_stencil<Stencil>( state );
    auto stencils  = get_stencils<Stencil>( state );

    int Na = state.n_cells[0];
    int Nb = state.n_cells[1];
    // int Nc = state.n_cells[2]; Not needed

    for( int i_cell = index; i_cell < state.n_cells_total; i_cell += stride )
    {
        int tupel[3];
        CUDA_HELPER::cu_tupel_from_idx( i_cell, tupel, state.n_cells, 3 ); // tupel now is {i, a, b, c}
        int a = tupel[0];
        int b = tupel[1];
        int c = tupel[2];

        for( int i_basis = 0; i_basis < state.n_cell_atoms; i_basis++ )
        {
            int idx_i = i_basis + state.n_cell_atoms * ( i_cell );

            // Allocate data for interacting spins
            Vector3 interaction_spins[N];

            for( int p = 0; p < N_Stencil; p++ )
            {
                Stencil & stencil = stencils[p];
                if( stencil.get_i() == i_basis )
                {
                    for( int idx_interaction = 0; idx_interaction < N; idx_interaction++ )
                    {
                        int idx_j
                            = stencil.get_j( idx_interaction )
                              + state.n_cell_atoms * ( ( a + stencil.get_da( idx_interaction ) ) + Na * ( b + stencil.get_db( idx_interaction ) + Nb * ( c + stencil.get_dc( idx_interaction ) ) ) );
                        if( idx_j >= 0 && idx_j < state.nos )
                        {
                            interaction_spins[idx_interaction] = state.spins[idx_j];
                        }
                        else
                        {
                            interaction_spins[idx_interaction] = { 0, 0, 0 };
                        }
                    }
                    state.gradient[idx_i] += stencil.gradient( state.spins[idx_i], interaction_spins );
                }
            }
        }
    }
}

// template for singe spin stencils (because size 0 plain arrays are not allowed)
template<typename Stencil>
__global__ void stencil_gradient( Device_State state )
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    auto N_Stencil = get_n_stencil<Stencil>( state );
    auto stencils  = get_stencils<Stencil>( state );

    for( int i_cell = index; i_cell < state.n_cells_total; i_cell += stride )
    {
        for( int i_basis = 0; i_basis < state.n_cell_atoms; i_basis++ )
        {
            int idx_i = i_basis + state.n_cell_atoms * ( i_cell );
            for( int p = 0; p < N_Stencil; p++ )
            {
                Stencil & stencil = stencils[p];
                if( stencil.get_i() == i_basis )
                {
                    state.gradient[idx_i] += stencil.gradient( state.spins[idx_i], {} );
                }
            }
        }
    }
}

__global__ void propagate_spins( Device_State state )
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for( int idx = index; idx < state.nos; idx += stride )
    {
        state.spins[idx] += state.timestep * state.gradient[idx];
        state.spins[idx].normalize();
    }
}

void iterate( Spirit::Host::Host_State & state, int N_iterations )
{
    int blockSize = 1024;
    int numBlocks = ( state.nos + blockSize - 1 ) / blockSize;

    Bfield_Stencil b( 0, {}, {}, {}, {}, { 0, 1, 0 } );
    Bfield_Stencil * b_dev_ptr;

    CUDA_HELPER::malloc_n( b_dev_ptr, 1 );
    CUDA_HELPER::copy_H2D( b_dev_ptr, &b );

    for( int iter = 0; iter < N_iterations; iter++ )
    {
        numBlocks = ( state.nos + blockSize - 1 ) / blockSize;
        set_gradient_zero<<<numBlocks, blockSize>>>( state.device_state );

        numBlocks = ( state.n_cells_total + blockSize - 1 ) / blockSize;

        stencil_gradient<1, ED_Stencil><<<numBlocks, blockSize>>>( state.device_state );
        stencil_gradient<Bfield_Stencil><<<numBlocks, blockSize>>>( state.device_state );
        stencil_gradient<K_Stencil><<<numBlocks, blockSize>>>( state.device_state );

        numBlocks = ( state.nos + blockSize - 1 ) / blockSize;
        propagate_spins<<<numBlocks, blockSize>>>( state.device_state );
        if( iter % 250 == 0 )
        {
            printf( "iter = %i\n", iter );
            state.Download();
            std::cout << "Spin[0,0,0] = " << state.spins[0].transpose() << "\n";
        }
    }
    cudaDeviceSynchronize();
    state.Download();
}

} // namespace Device
} // namespace Spirit
#endif