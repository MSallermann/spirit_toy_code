#include "State.h"
#include "Api_State.hpp"

State * State_Setup( int n_cells[3], int n_cell_atoms, bool boundary_conditions[3] ) noexcept
{
    return new State( n_cells, n_cell_atoms, boundary_conditions );
}

void State_Download( State * state ) noexcept
{
    state->core_state.download();
}

void State_Upload( State * state ) noexcept
{
    state->core_state.upload();
}

void State_Allocate( State * state ) noexcept
{
    state->core_state.upload();
}

void State_Delete( State * state ) noexcept
{
    delete state;
}