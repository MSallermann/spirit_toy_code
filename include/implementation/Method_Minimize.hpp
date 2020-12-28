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
    Method_Minimize( Interface::State * state ) : Method_Implementation( state )
    {
        this->eligible_solvers.insert( Interface::SolverType::Gradient_Descent );
    }

    void iterate( int N_iterations ) override
    {

        for( int iter = 0; iter < N_iterations; iter++ )
        {
            Kernels::set_gradient_zero( m_state->gradient.data(), m_state_host->geometry );
            m_state_host->hamiltonian_device->get_gradient( m_state->gradient.data(), m_state->spins.data(), m_state_host->geometry );
            m_solver->progagate_spins( m_state_host );

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

} // namespace Implementation
} // namespace Spirit

#endif