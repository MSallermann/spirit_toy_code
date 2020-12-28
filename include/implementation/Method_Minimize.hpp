#pragma once
#ifndef IMPLEMENTATION_METHOD_MINIMIZE_HPP
#define IMPLEMENTATION_METHOD_MINIMIZE_HPP
#include "implementation/Fields.hpp"
#include "implementation/Hamiltonian.hpp"
#include "implementation/Kernels.hpp"
#include "interface/Method.hpp"
#include <iostream>
namespace Spirit
{
namespace Implementation
{

class Method_Minimize : public Interface::Method_Implementation
{
public:
    Method_Minimize( Interface::State & state ) : Method_Implementation( state )
    {
        this->eligible_solvers.insert( Interface::SolverType::Gradient_Descent );
    }

    void iterate( int N_iterations ) override
    {

        for( int iter = 0; iter < N_iterations; iter++ )
        {
            Kernels::set_gradient_zero( state.fields->gradient.data(), state.geometry );
            state.hamiltonian_device->get_gradient( state.fields->gradient.data(), state.fields->spins.data(), state.geometry );
            solver->progagate_spins( state );

            if( iter % 250 == 0 )
            {
                state.download();
                printf( "iter = %i\n", iter );
                std::cout << "    spin[0,0,0]     = " << state.spins[0].transpose() << "\n";
                std::cout << "    gradient[0,0,0] = " << state.gradient[0].transpose() << "\n";
            }
        }
    }
};

} // namespace Implementation
} // namespace Spirit

#endif