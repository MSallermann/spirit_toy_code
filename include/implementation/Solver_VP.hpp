#pragma once
#ifndef IMPLEMENTATION_SOLVER_VP
#define IMPLEMENTATION_SOLVER_VP
#include "implementation/Fields.hpp"
#include "implementation/Kernels.hpp"
#include "implementation/Lambda.hpp"

#include "interface/Method.hpp"
#include "interface/State.hpp"

namespace Spirit
{
namespace Implementation
{

class Solver_VP : public Interface::Solver_Implementation
{
    device_vector<Vector3> velocities;
    device_vector<Vector3> velocities_previous;
    device_vector<Vector3> forces_previous;
    device_vector<scalar> dot;
    device_vector<scalar> force_norm2_vec;
    device_type<scalar> projection;
    device_type<scalar> force_norm2;
    device_type<scalar> ratio;
    scalar m;
    Kernels::Summator summator;

public:
    Solver_VP( Interface::State & state )
            : Interface::Solver_Implementation( state ),
              velocities( state.geometry.nos, { 0, 0, 0 } ),
              velocities_previous( state.geometry.nos, { 0, 0, 0 } ),
              forces_previous( state.geometry.nos, { 0, 0, 0 } ),
              dot( state.geometry.nos, 0 ),
              force_norm2_vec( state.geometry.nos, 0 ),
              projection( 0 ),
              force_norm2( 0 ),
              summator( Kernels::Summator( state.geometry.nos ) ),
              m( 1e3 ){};

    virtual void progagate_spins( Interface::State & state ) override
    {
        auto velocity   = velocities.data();
        auto force      = state.fields->gradient.data();
        auto force_prev = forces_previous.data();
        auto m          = this->m;
        scalar * d      = dot.data();
        scalar * fnorm2 = force_norm2_vec.data();

        // Calculate the new velocity

        Lambda::apply( state.geometry.nos, [force, force_prev, velocity, m, d, fnorm2] SPIRIT_LAMBDA( int idx ) {
            velocity[idx] += 0.5 / m * ( force_prev[idx] + force[idx] );
            d[idx]      = velocity[idx].dot( force[idx] );
            fnorm2[idx] = force[idx].squaredNorm();
        } );

        // Get the projection of the velocity on the force
        summator.sum( projection.data(), dot.data() );
        // Get the squared norm of the force
        summator.sum( force_norm2.data(), force_norm2_vec.data() );

        Lambda::apply( state.geometry.nos, [velocity, force, f2 = force_norm2.data(), p = projection.data()] SPIRIT_LAMBDA( int idx ) {
            scalar ratio = *p / *f2;
            if( *p <= 0 )
            {
                velocity[idx] = { 0, 0, 0 };
            }
            else
            {
                velocity[idx] = force[idx] * ratio;
            }
        } );

        // clang-format off
        Lambda::apply(
            state.geometry.nos, [conf = state.fields->spins.data(), dt = state.solver_parameters.timestep, m, velocity, force] SPIRIT_LAMBDA(
                                    int idx ) 
                                    { conf[idx] = ( conf[idx] + dt * velocity[idx] + 0.5 / m * dt * force[idx] ).normalized(); } );
        // clang-format on

        // Save force to previous
        forces_previous     = state.fields->gradient;
        velocities_previous = velocities;
    }
};

} // namespace Implementation
} // namespace Spirit

#endif
