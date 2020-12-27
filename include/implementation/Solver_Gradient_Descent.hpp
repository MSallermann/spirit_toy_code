#pragma once
#ifndef IMPLEMENTATION_SOLVER_GRADIENT_DESCENT_HPP
#define IMPLEMENTATION_SOLVER_GRADIENT_DESCENT_HPP
#include "implementation/State.hpp"
#include "interface/Method.hpp"

namespace Spirit
{
namespace Device
{

class Solver_Gradient_Descent : public Solver_Implementation
{
    virtual void progagate_spins( Device::State * state ) override
    {
        printf( "progagate spins gd\n" );
    };
};

} // namespace Device
} // namespace Spirit

#endif
