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
    device_type<scalar> projection;
    device_type<scalar> force_norm2;
    device_type<scalar> ratio;
    scalar m;
    Kernels::Summator summator;

public:
    Solver_VP( Interface::State & state )
            : Interface::Solver_Implementation( state ),
              velocities( state.geometry.nos ),
              velocities_previous( state.geometry.nos, { 0, 0, 0 } ),
              forces_previous( state.geometry.nos, { 0, 0, 0 } ),
              dot( state.geometry.nos ),
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

        // Calculate the new velocity
        Lambda::apply( state.geometry.nos, [force, force_prev, velocity, m] SPIRIT_LAMBDA( int idx ) {
            velocity[idx] += 0.5 / m * ( force_prev[idx] + force[idx] );
        } );

        // Get the projection of the velocity on the force
        Kernels::dot( dot.data(), velocity, force, state.geometry.nos );
        summator.sum( projection.data(), dot.data() );

        // Get the squared norm of the force
        Kernels::dot( dot.data(), force, force, state.geometry.nos );
        summator.sum( force_norm2.data(), dot.data() );

        Lambda::apply( 1, [r = ratio.data(), f = force_norm2.data(), p = projection.data()] SPIRIT_LAMBDA( int idx ) { r[0] = p[0] / f[0]; } );

        Lambda::apply( state.geometry.nos, [velocity, force, r = ratio.data(), p = projection.data()] SPIRIT_LAMBDA( int idx ) {
            if( p[0] <= 0 )
            {
                velocity[idx] = { 0, 0, 0 };
            }
            else
            {
                velocity[idx] = force[idx] * r[0];
            }
        } );

        // clang-format off
        Lambda::apply(
            state.geometry.nos, [conf = state.fields->spins.data(), dt = state.solver_parameters.timestep, m, velocity, force] SPIRIT_LAMBDA(
                                    int idx ) 
                                    { conf[idx] = ( conf[idx] + dt * velocity[idx] + 0.5 / m * dt * force[idx] ).normalized(); } );
        // clang-format on

        // Save force to previous
        auto force_pr    = forces_previous.data();
        auto velocity_pr = velocities_previous.data();

        forces_previous     = state.fields->gradient;
        velocities_previous = velocities;
    }
};

} // namespace Implementation
} // namespace Spirit

#endif
