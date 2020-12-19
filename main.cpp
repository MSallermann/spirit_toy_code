#include "backend.hpp"
#include <chrono>
#include <iostream>

constexpr int N_iterations = 500;

int main( int argc, char * argv[] )
{
    std::array<int, 5> cmd_args = { 100, 100, 1, 1, 500 };
    if( argc > 1 )
    {
        for( int i = 1; i < argc; i++ )
            try
            {
                cmd_args[i - 1] = std::stoi( argv[i] );
            }
            catch( ... )
            {
                std::cout << "Could not parse command line arg no. " << i << ": '" << argv[i] << "', using default " << cmd_args[i - 1] << "\n";
            };
    }
    int Na = cmd_args[0], Nb = cmd_args[1], Nc = cmd_args[2], Nbasis = cmd_args[3], N_iterations = cmd_args[4];
    printf( "Using Na = %i, Nb = %i, Nc = %i, Nbasis = %i, Niteration(s) = %i\n", cmd_args[0], cmd_args[1], cmd_args[2], cmd_args[3], cmd_args[4] );
    std::cout << "Backend: " << description() << "\n";
    State state = State( { Na, Nb, Nc }, Nbasis, 1e-3 );

    std::vector<Pair_Stencil> stencils;

    Pair_Stencil temp;
    temp.i      = 0;
    temp.j      = 0;
    temp.da     = 1;
    temp.db     = 2;
    temp.dc     = 3;
    temp.matrix = 1 * Matrix3::Identity();

    for( int i = 0; i < 10; i++ )
        stencils.push_back( temp );

    state.Set_Pair_Stencils( stencils );

    auto start = std::chrono::high_resolution_clock::now();
    iterate( state.backend, N_iterations );
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "Elapsed time: " << elapsed_seconds.count() * 1e3 << " ms (" << N_iterations << " iterations)\n";
    std::cout << "              " << elapsed_seconds.count() / N_iterations * 1e3 << " ms per iteration\n";
    std::cout << "              " << N_iterations / elapsed_seconds.count() << " iterations per second\n";
}