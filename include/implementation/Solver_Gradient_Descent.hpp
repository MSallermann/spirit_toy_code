#pragma once
#ifndef IMPLEMENTATION_SOLVER_GRADIENT_DESCENT_HPP
#define IMPLEMENTATION_SOLVER_GRADIENT_DESCENT_HPP
#include "implementation/Kernels.hpp"
#include "implementation/State.hpp"
#include "interface/Method.hpp"

namespace Spirit
{
namespace Implementation
{

class Solver_Gradient_Descent : public Solver_Implementation
{
    virtual void progagate_spins( Implementation::State * state ) override
    {
        Kernels::propagate_spins( state->spins.data(), state->gradient.data(), state->pod );
    };
};

} // namespace Implementation
} // namespace Spirit

#endif
