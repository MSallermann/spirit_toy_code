// #include "implementation/Host_State.hpp"
// #include "implementation/backend.hpp"
#include <chrono>
#include <iostream>

#include "interface/Method.hpp"
#include "interface/State.hpp"
#include "interface/Stencil.hpp"

using namespace Spirit;
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

    std::array<int, 3> n_cells = { Na, Nb, Nc };
    int n_cell_atoms           = Nbasis;
    Spirit::Interface::State s = Spirit::Interface::State( n_cells, n_cell_atoms );

    using EDt = Interface::Stencil<2, Matrix3>;
    std::vector<EDt> ed_stencils;

    Matrix3 matrix;
    matrix << 1, 0, 1, 0, 1, 0, -1, 0, 1;
    ed_stencils.push_back( EDt( 0, { 0 }, { 1 }, { 0 }, { 0 }, matrix ) );
    ed_stencils.push_back( EDt( 0, { 0 }, { -1 }, { 0 }, { 0 }, matrix ) );
    ed_stencils.push_back( EDt( 0, { 0 }, { 0 }, { 1 }, { 0 }, matrix ) );
    ed_stencils.push_back( EDt( 0, { 0 }, { 0 }, { -1 }, { 0 }, matrix ) );
    ed_stencils.push_back( EDt( 0, { 0 }, { 0 }, { 0 }, { 1 }, matrix ) );
    ed_stencils.push_back( EDt( 0, { 0 }, { 0 }, { 0 }, { -1 }, matrix ) );
    s.hamiltonian.ed_stencils = ed_stencils;

    using Kt = Interface::Stencil<1, Vector3>;
    std::vector<Kt> k_stencils;
    Vector3 vec = Vector3::Identity();
    k_stencils.push_back( Kt( 0, {}, {}, {}, {}, vec ) );
    s.hamiltonian.k_stencils = k_stencils;

    std::vector<Kt> b_stencils;
    b_stencils.push_back( Kt( 0, {}, {}, {}, {}, vec ) );
    s.hamiltonian.b_stencils = b_stencils;
    s.set_domain( { 2, 2, 2 } );

    s.allocate();
    s.upload();

    Interface::Method_Implementation * method = get_method_implementation( &s, Interface::MethodType::Minimisation );
    Interface::Solver_Implementation * solver = get_solver_implementation( &s, Interface::SolverType::Gradient_Descent );

    Interface::Method m( method );
    m.set_solver( solver );

    std::cout << "Sart Iterations\n";

    auto start = std::chrono::high_resolution_clock::now();
    m.iterate( N_iterations );
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "End Iterations\n";
    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "Elapsed time: " << elapsed_seconds.count() * 1e3 << " ms (" << N_iterations << " iterations)\n";
    std::cout << "              " << elapsed_seconds.count() / N_iterations * 1e3 << " ms per iteration\n";
    std::cout << "              " << N_iterations / elapsed_seconds.count() << " iterations per second\n";

    return 0;
}