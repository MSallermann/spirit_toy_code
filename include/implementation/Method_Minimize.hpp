#pragma once
#ifndef IMPLEMENTATION_METHOD_MINIMIZE_HPP
#define IMPLEMENTATION_METHOD_MINIMIZE_HPP
#include "implementation/State.hpp"
#include "interface/Method.hpp"
#include <iostream>
namespace Spirit
{
namespace Device
{

class Method_Minimize : public Method_Implementation
{
    State * state;

public:
    Method_Minimize( Host::State * state ) : Method_Implementation( state )
    {
        this->eligible_solvers.insert( SolverType::Gradient_Descent );
    }

    void iterate( int N_iterations ) override
    {
        std::cout << "iterate minimize\n";
        m_solver->progagate_spins( state );
    }
};

} // namespace Device
} // namespace Spirit

#endif