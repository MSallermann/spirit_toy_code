#include "catch.hpp"
#include "implementation/Kernels.hpp"
#include "implementation/Memory.hpp"
#include <iostream>
#include <vector>

using namespace Spirit;

TEST_CASE( "Kernels" )
{
    Implementation::device_vector<scalar> s1_d( 100, 0.33 );
    Implementation::device_type<scalar> res_d;

    Implementation::Kernels::Summator summator( s1_d.size() );
    summator.sum( res_d.data(), s1_d.data() );
    REQUIRE( Approx( res_d.download() ) == 100 * 0.33 );

    Implementation::device_vector<Vector3> v1_d( 100 );
    Implementation::device_vector<Vector3> v2_d( 100 );
    std::vector<Vector3> v1( 100, { 1, 1, 1 } );
    std::vector<Vector3> v2( 100, { 2, 2, 2 } );
    v1_d.copy_from( v1 );
    v2_d.copy_from( v2 );

    Implementation::device_vector<scalar> dot_temp( 100 );

    Implementation::Kernels::dot( dot_temp.data(), v1_d.data(), v2_d.data(), v1_d.size() );
    summator.sum( res_d.data(), dot_temp.data() );
    REQUIRE( Approx( res_d.download() ) == 100 * 6 );
}