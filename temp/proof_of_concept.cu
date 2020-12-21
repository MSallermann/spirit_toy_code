// For CPU
// using scalar = double;
// #define SPIRIT_LAMBDA
// #define STATE_ATTRIBUTE

// For GPU
using scalar = float;
#define SPIRIT_LAMBDA __host__ __device__
#define STATE_ATTRIBUTE __device__

#include "Eigen/Dense"
#include <stdarg.h>
#include <array>
#include <chrono>
#include <iostream>
#include <vector>

using Matrix3 = Eigen::Matrix<scalar, 3, 3>;
using Vector3 = Eigen::Matrix<scalar, 3, 1>;

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

namespace spirit
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
void cfree( T dev_ptr )
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

template<int N, typename PARAM>
struct Stencil
{
    using Vector3NArray               = Vector3[N];
    using t_gradient_func             = Vector3( Vector3, Vector3[N], PARAM );
    using t_energy_func               = scalar( Vector3, Vector3[N], PARAM );
    using t_energy_from_gradient_func = scalar( Vector3, Vector3, Vector3[N], PARAM );

    int i;
    std::array<int, N> j;
    std::array<int, N> da;
    std::array<int, N> db;
    std::array<int, N> dc;

    PARAM param;

    Stencil(){};
    Stencil( int i, PARAM param ) : i( i ), param( param ){};
    Stencil( int i, std::array<int, N> j, std::array<int, N> da, std::array<int, N> db, std::array<int, N> dc, PARAM param ) : i( i ), j( j ), da( da ), db( db ), dc( dc ), param( param ){};

    STATE_ATTRIBUTE
    Vector3 gradient( const Vector3 & spin, const Vector3NArray & interaction_spins );

    STATE_ATTRIBUTE
    scalar energy( const Vector3 & spin, const Vector3NArray & interaction_spins );

    STATE_ATTRIBUTE
    scalar energy_from_gradient( const Vector3 & spin, const Vector3 & gradient, const Vector3NArray & interaction_spins )
    {
        return energy( spin, interaction_spins );
    };

    int product( int u, int v )
    {
        return u * v;
    }

    STATE_ATTRIBUTE
    int get_i()
    {
        return i;
    };

    STATE_ATTRIBUTE
    int get_j( int idx )
    {
        return ( (int *)&( this->j ) )[idx];
    };

    STATE_ATTRIBUTE
    int get_da( int idx )
    {
        return ( (int *)&( this->da ) )[idx];
    };

    STATE_ATTRIBUTE
    int get_db( int idx )
    {
        return ( (int *)&( this->db ) )[idx];
    };

    STATE_ATTRIBUTE
    int get_dc( int idx )
    {
        return ( (int *)&( this->db ) )[idx];
    };
};

struct ED_Stencil : Stencil<1, Matrix3>
{
    ED_Stencil( int i, std::array<int, 1> j, std::array<int, 1> da, std::array<int, 1> db, std::array<int, 1> dc, Matrix3 param ) : Stencil<1, Matrix3>( i, j, da, db, dc, param ){};

    STATE_ATTRIBUTE
    Vector3 gradient( const Vector3 & spin, const Vector3NArray & interaction_spins )
    {
        return param * interaction_spins[0];
    }

    STATE_ATTRIBUTE
    scalar energy( const Vector3 & spin, const Vector3NArray & interaction_spins )
    {
        return spin.transpose() * param * interaction_spins[0];
    }
};

struct Backend_State
{
    bool allocated = false;
    int nos;
    int n_cell_atoms;
    int n_cells_total;
    int n_cells[3];
    scalar timestep;

    Vector3 * spins;
    Vector3 * gradient;

    int n_ed;
    ED_Stencil * ed_stencils;

    void free()
    {
        if( !allocated )
            return;

        std::cout << "Freeing backend resources\n";
        cudaFree( spins );
        cudaFree( gradient );
        cudaFree( ed_stencils );
    }
};

class Host_State
{
public:
    // Host Stack memory
    int nos;
    int n_cell_atoms;
    int n_cells_total;
    std::array<int, 3> n_cells;
    scalar timestep = 1e-3;

    // Host Heap memory
    std::vector<Vector3> spins;
    std::vector<Vector3> gradient;
    std::vector<ED_Stencil> ed_stencils;

    // Device pointer to backend_state
    Backend_State backend_state;

    Host_State( std::array<int, 3> n_cells, int n_cell_atoms, const std::vector<ED_Stencil> & ed_stencils ) : n_cells( n_cells ), n_cell_atoms( n_cell_atoms ), ed_stencils( ed_stencils )
    {
        timestep      = 1e-3;
        n_cells_total = n_cells[0] * n_cells[1] * n_cells[2];
        nos           = n_cell_atoms * n_cells_total;
        gradient      = std::vector<Vector3>( nos, { 0, 0, 0 } );
        spins         = std::vector<Vector3>( nos, { 1, 1, 1 } );
        Allocate_GPU_Backend();
        Upload();
    }

    void Set_Domain( const Vector3 & vec )
    {
#pragma omp parallel for
        for( int i = 0; i < nos; i++ )
        {
            spins[i] = vec;
        }
        Upload();
    }

    void Download()
    {
        copy_vector_D2H( spins, backend_state.spins );
        copy_vector_D2H( gradient, backend_state.gradient );
    }

    void Upload()
    {
        copy_vector_H2D( backend_state.spins, spins );
        copy_vector_H2D( backend_state.ed_stencils, ed_stencils );
    }

    void Allocate_GPU_Backend()
    {
        backend_state.free();
        backend_state               = Backend_State();
        backend_state.allocated     = true;
        backend_state.nos           = nos;
        backend_state.n_cells_total = n_cells_total;
        backend_state.n_cell_atoms  = n_cell_atoms;
        backend_state.n_cells[0]    = n_cells[0];
        backend_state.n_cells[1]    = n_cells[1];
        backend_state.n_cells[2]    = n_cells[2];
        backend_state.n_ed          = ed_stencils.size();
        backend_state.timestep      = timestep;
        malloc_n( backend_state.gradient, nos );
        malloc_n( backend_state.spins, nos );
        malloc_n( backend_state.ed_stencils, ed_stencils.size() );
    }

    ~Host_State()
    {
        backend_state.free();
    }
};

template<int N, typename Stencil>
__global__ void stencil_gradient( Backend_State state, Stencil * stencils, int N_Stencil )
{

    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int Na            = state.n_cells[0];
    int Nb            = state.n_cells[1];
    int Nc            = state.n_cells[2];
    int N_cells_total = Na * Nb * Nc;

    for( int i = index; i < state.nos; i += stride )
    {
        state.gradient[i] = { 0, 0, 0 };
    }
    // return;
    for( int i_cell = index; i_cell < state.n_cells_total; i_cell += stride )
    {
        int tupel[3];
        cu_tupel_from_idx( i_cell, tupel, state.n_cells, 3 ); // tupel now is {i, a, b, c}
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
                              + state.n_cell_atoms * ( ( a + stencil.get_da( idx_interaction ) ) + Na * ( b + stencil.get_db( idx_interaction ) + Nc * ( c + stencil.get_dc( idx_interaction ) ) ) );
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

__global__ void propagate_spins( Backend_State state )
{
    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for( int idx = index; idx < state.nos; idx += stride )
    {
        state.spins[idx] += state.timestep * state.gradient[idx];
        state.spins[idx].normalize();
    }
}

void iterate( Host_State & state, int N_iterations )
{

    int blockSize    = 1024;
    int numBlocks    = ( state.nos + blockSize - 1 ) / blockSize;
    auto ed_gradient = [] __device__( const Vector3 & spin, Vector3 interaction_spins[3], Matrix3 param ) { return param * spin; }; // Not cool

    for( int iter = 0; iter < N_iterations; iter++ )
    {
        stencil_gradient<1, ED_Stencil><<<numBlocks, blockSize>>>( state.backend_state, state.backend_state.ed_stencils, state.backend_state.n_ed );
        propagate_spins<<<numBlocks, blockSize>>>( state.backend_state );
        if( iter % 100 == 0 )
        {
            printf( "iter = %i\n", iter );
            state.Download();
            std::cout << "Spin[0,0,0] = " << state.spins[0].transpose() << "\n";
        }
    }
    cudaDeviceSynchronize();
    state.Download();
}

} // namespace spirit

using namespace spirit;
int main( void )
{
    std::vector<ED_Stencil> stencils;

    Matrix3 matrix;
    matrix << 1, 0, 1, 0, 1, 0, -1, 0, 1;
    stencils.push_back( ED_Stencil( 0, { 0 }, { 1 }, { 0 }, { 0 }, matrix ) );
    stencils.push_back( ED_Stencil( 0, { 0 }, { -1 }, { 0 }, { 0 }, matrix ) );
    stencils.push_back( ED_Stencil( 0, { 0 }, { 0 }, { 1 }, { 0 }, matrix ) );
    stencils.push_back( ED_Stencil( 0, { 0 }, { 0 }, { -1 }, { 0 }, matrix ) );
    stencils.push_back( ED_Stencil( 0, { 0 }, { 0 }, { 0 }, { 1 }, matrix ) );
    stencils.push_back( ED_Stencil( 0, { 0 }, { 0 }, { 0 }, { -1 }, matrix ) );

    int Na = 100, Nb = 100, Nc = 100, Nbasis = 1, N_iterations = 20000;
    Host_State state( { Na, Nb, Nc }, Nbasis, stencils );
    state.Set_Domain( { 2, 2, 2 } );
    int blockSize = 1024;
    int numBlocks = ( state.n_cells_total + blockSize - 1 ) / blockSize;

    printf( "Using Na = %i, Nb = %i, Nc = %i, Nbasis = %i, Niteration(s) = %i\n", Na, Nb, Nc, Nbasis, N_iterations );
    std::cout << "Sart Iterations\n";
    auto start = std::chrono::high_resolution_clock::now();
    iterate( state, N_iterations );
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "End Iterations\n";
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "Elapsed time: " << elapsed_seconds.count() * 1e3 << " ms (" << N_iterations << " iterations)\n";
    std::cout << "              " << elapsed_seconds.count() / N_iterations * 1e3 << " ms per iteration\n";
    std::cout << "              " << N_iterations / elapsed_seconds.count() << " iterations per second\n";
    // stencil_gradient<<<blockSize, numBlocks>>>( state.backend_state, state.backend_state->ed_stencils, state.backend_state->n_ED );
};