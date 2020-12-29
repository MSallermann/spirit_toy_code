#pragma once
#ifndef INTERFACE_METHOD_HPP
#define INTERFACE_METHOD_HPP
#include "interface/State.hpp"
#include <iostream>
#include <memory>
#include <unordered_set>

namespace Spirit
{
namespace Interface
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
    Interface::State & state;

public:
    SolverType m_type;
    Solver_Implementation( Interface::State & state ) : state( state ) {}
    virtual void progagate_spins( Interface::State & state ) = 0;
    virtual ~Solver_Implementation(){};
};

class Method_Implementation
{
protected:
    int m_iteration_step;
    MethodType m_type;
    Solver_Implementation * solver = nullptr;
    std::unordered_set<SolverType> eligible_solvers;

    Interface::State & state;

public:
    virtual ~Method_Implementation()
    {
        if( solver != nullptr )
        {
            delete solver;
        }
    }

    Method_Implementation( Interface::State & state ) : state( state ){};
    virtual void iterate( int N_iterations ) = 0;
    // virtual void iteration()                              = 0;
    // virtual void compose_iterations( int iteration_step ) = 0;

    virtual void set_solver( Solver_Implementation * solver )
    {
        this->solver = solver;

        // The following causes error on windows for some reason (find out why):
        // if( eligible_solvers.find( solver->m_type ) != eligible_solvers.end() )
        // this->solver = solver;
        // else
        // throw std::runtime_error( "Ineligible solver" );
    }
};

class Method
{
protected:
    Method_Implementation * m_implementation = nullptr;

public:
    Method( Method_Implementation * implementation ) : m_implementation( implementation ) {}

    virtual ~Method()
    {
        if( m_implementation != nullptr )
        {
            delete m_implementation;
        }
    }

    virtual void set_solver( Solver_Implementation * solver )
    {
        this->m_implementation->set_solver( solver );
    }

    virtual void iterate( int N_iterations )
    {
        this->m_implementation->iterate( N_iterations );
    }
};

Solver_Implementation * get_solver_implementation( Interface::State & state, SolverType type );
Method_Implementation * get_method_implementation( Interface::State & state, MethodType type );

} // namespace Interface
} // namespace Spirit
#endif