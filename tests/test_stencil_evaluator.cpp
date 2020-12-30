#include "Definitions.hpp"
#include "catch.hpp"
#include "implementation/Kernels.hpp"
#include "implementation/Memory.hpp"
#include "implementation/Stencil_Evaluator.hpp"
#include "implementation/Stencil_Terms.hpp"

#include <iostream>
#include <vector>

using namespace Spirit;
using namespace Spirit::Implementation;

class Test_Stencil : public StencilImp<2, scalar>
{
public:
    HD_ATTRIBUTE
    Vector3 gradient( const Vector3NArray & interaction_spins )
    {
        return interaction_spins[0] + param * interaction_spins[1];
    }
};

TEST_CASE( "Stencil_Evaluator" )
{
    Test_Stencil h_test_stencil;
    h_test_stencil.i     = 0;
    h_test_stencil.j     = { 0 };
    h_test_stencil.da    = { 1 };
    h_test_stencil.db    = { 1 };
    h_test_stencil.dc    = { 0 };
    h_test_stencil.param = 1.7;

    device_type<Test_Stencil> test_stencil( h_test_stencil );

    // std::cout << test_stencil.data()->param << "\n";

    Interface::State::Geometry geom( { 30, 30, 30 }, 1, { true, false, true } );
    Vector3 s = { 1, 1, 1 };
    device_vector<Vector3> spins( geom.nos, s );
    device_vector<Vector3> gradient( geom.nos, { 0, 0, 0 } );
    std::vector<Vector3> host_gradient( geom.nos );
    std::vector<Vector3> host_spins( geom.nos );

    stencil_gradient( gradient.data(), spins.data(), geom, 1, test_stencil.data() );

#ifdef BACKEND_CUDA
    cudaDeviceSynchronize();
#endif
    gradient.copy_to( host_gradient );
    spins.copy_to( host_spins );

    for( int a = 0; a < geom.n_cells[0]; a++ )
        for( int b = 0; b < geom.n_cells[1]; b++ )
            for( int c = 0; c < geom.n_cells[2]; c++ )
            {
                int i = a + geom.n_cells[0] * ( b + geom.n_cells[1] * c );
                INFO( "i = " << i );
                INFO( "host_gradient[i] = " << host_gradient[i].transpose() );
                Vector3 expected;
                if( b == geom.n_cells[1] - 1 ) // at towards the side b we expecpt a different result because of open boundary conditions and db=1
                {
                    expected = s;
                }
                else
                {
                    expected = ( s + h_test_stencil.param * s );
                }
                INFO( "expected         = " << expected.transpose() );
                REQUIRE( host_gradient[i].isApprox( expected ) );
            }
}