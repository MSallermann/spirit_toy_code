#pragma once
#ifndef IMPLEMENTATION_METHOD_MINIMIZE_HPP
#define IMPLEMENTATION_METHOD_MINIMIZE_HPP
#include "implementation/Kernels.hpp"
#include "implementation/State.hpp"
#include "interface/Method.hpp"
#include <iostream>
namespace Spirit
{
namespace Device
{

class Method_Minimize : public Method_Implementation
{
public:
    Method_Minimize( Host::State * state ) : Method_Implementation( state )
    {
        this->eligible_solvers.insert( SolverType::Gradient_Descent );
    }

    void iterate( int N_iterations ) override
    {

        for( int iter = 0; iter < N_iterations; iter++ )
        {
            Kernels::set_gradient_zero( m_state );
            m_state->Gradient_Async( m_state->gradient );
            m_solver->progagate_spins( m_state );

            if( iter % 250 == 0 )
            {
                m_state_host->download();
                printf( "iter = %i\n", iter );
                std::cout << "    spin[0,0,0]     = " << m_state_host->spins[0].transpose() << "\n";
                std::cout << "    gradient[0,0,0] = " << m_state_host->gradient[0].transpose() << "\n";
            }
        }
    }
};

} // namespace Device
} // namespace Spirit

#endif