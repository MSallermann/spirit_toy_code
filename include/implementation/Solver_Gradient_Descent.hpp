#pragma once
#ifndef IMPLEMENTATION_SOLVER_GRADIENT_DESCENT_HPP
#define IMPLEMENTATION_SOLVER_GRADIENT_DESCENT_HPP
#include "implementation/Fields.hpp"
#include "implementation/Kernels.hpp"
#include "interface/Method.hpp"
#include "interface/State.hpp"

namespace Spirit
{
namespace Implementation
{

class Solver_Gradient_Descent : public Interface::Solver_Implementation
{
public:
    Solver_Gradient_Descent( Interface::State & state ) : Interface::Solver_Implementation( state ){};
    virtual void progagate_spins( Interface::State & state ) override
    {
        Kernels::propagate_spins( state.fields->spins.data(), state.fields->gradient.data(), state.solver_parameters, state.geometry );
    };
};

} // namespace Implementation
} // namespace Spirit

#endif
