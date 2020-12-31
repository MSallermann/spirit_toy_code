#include "Method.h"
#include "Api_State.hpp"
#include "interface/Method.hpp"
#include "interface/State.hpp"

void Set_Method( State * state, Method type, Solver solver ) noexcept
{
    state->method = Spirit::Interface::Method( state->core_state, Spirit::Interface::MethodType( type ) );
    state->method.set_solver( Spirit::Interface::SolverType( solver ) );
}

void Iterate( State * state, int N_iterations ) noexcept
{
    state->method.iterate( N_iterations );
}