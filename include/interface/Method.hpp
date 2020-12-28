#pragma once
#ifndef INTERFACE_METHOD_HPP
#define INTERFACE_METHOD_HPP
#include "interface/State.hpp"
#include <iostream>
#include <memory>
#include <unordered_set>

namespace Spirit
{

enum MethodType
{
    Minimisation = 0,
    LLG          = 1
};

enum SolverType
{
    Gradient_Descent = 0,
    VP               = 1
};

class Solver_Implementation
{
public:
    SolverType m_type;
    virtual void progagate_spins( Implementation::State * state ) = 0;
};

class Method_Implementation
{
protected:
    int m_iteration_step;
    MethodType m_type;
    Solver_Implementation * m_solver;
    std::unordered_set<SolverType> eligible_solvers;

    Interface::State * m_state_host;
    Implementation::State * m_state;

public:
    virtual ~Method_Implementation() {}
    Method_Implementation( Interface::State * state ) : m_state_host( state ), m_state( state->device_state ){};
    virtual void iterate( int N_iterations ) = 0;
    virtual void set_solver( Solver_Implementation * solver )
    {
        if( eligible_solvers.find( solver->m_type ) != eligible_solvers.end() )
            m_solver = solver;
        else
            throw std::runtime_error( "Ineligible solver" );
    }
    // virtual void iteration()                              = 0;
    // virtual void compose_iterations( int iteration_step ) = 0;
};

class Method
{
protected:
    Method_Implementation * m_implementation = nullptr;

public:
    Method( Method_Implementation * implementation ) : m_implementation( implementation ) {}

    virtual ~Method() {}

    virtual void set_solver( Solver_Implementation * solver )
    {
        this->m_implementation->set_solver( solver );
    }

    virtual void iterate( int N_iterations )
    {
        this->m_implementation->iterate( N_iterations );
    }
};

Solver_Implementation * get_solver_implementation( Interface::State * state, SolverType type );
Method_Implementation * get_method_implementation( Interface::State * state, MethodType type );

} // namespace Spirit
#endif