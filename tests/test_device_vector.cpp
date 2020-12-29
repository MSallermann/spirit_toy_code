#include "Definitions.hpp"
#include "catch.hpp"
#include "implementation/Memory.hpp"

#include <iostream>
#include <vector>

using namespace Spirit;
using namespace Spirit::Implementation;

TEST_CASE( "Device_Vector" )
{
    int n = 100;
    std::vector<Vector3> spins( n, { 0, 2, 0 } );
    device_vector<Vector3> spins_d( spins.size(), { 2, 0, 2 } );
    device_vector<Vector3> spins_d_2( spins.size(), { 0, 0, 0 } );

    spins_d.copy_from( spins );
    spins_d_2 = spins_d;

    for( auto & s : spins )
    {
        s = { -1, -1, -1 };
    }

    spins_d.copy_to( spins );

    for( auto & s : spins )
    {
        REQUIRE( ( s - Vector3{ 0, 2, 0 } ).norm() == 0 );
    }
}