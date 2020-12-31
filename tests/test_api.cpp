#include "Hamiltonian.h"
#include "Method.h"
#include "State.h"
#include "catch.hpp"
#include <array>
#include <iostream>

TEST_CASE( "C-API" )
{
    int Na = 100, Nb = 100, Nc = 1, Nbasis = 1, N_iterations = 100;
    printf( "Using Na = %i, Nb = %i, Nc = %i, Nbasis = %i, Niteration(s) = %i\n", Na, Nb, Nc, Nbasis, N_iterations );

    int n_cell[3]               = { Na, Nb, Nc };
    bool boundary_conditions[3] = { true, true, true };
    State * p_state             = State_Setup( n_cell, Nbasis, boundary_conditions );

    Push_ED_Stencil( p_state, ED_Stencil{ 0, { 0 }, { 1 }, { 0 }, { 0 }, 1, 1 } );
    Push_ED_Stencil( p_state, ED_Stencil{ 0, { 0 }, { -1 }, { 0 }, { 0 }, 1, 1 } );
    Push_ED_Stencil( p_state, ED_Stencil{ 0, { 0 }, { 0 }, { 1 }, { 0 }, 1, 1 } );
    Push_ED_Stencil( p_state, ED_Stencil{ 0, { 0 }, { 0 }, { -1 }, { 0 }, 1, 1 } );
    Push_ED_Stencil( p_state, ED_Stencil{ 0, { 0 }, { 0 }, { 0 }, { 1 }, 1, 1 } );
    Push_ED_Stencil( p_state, ED_Stencil{ 0, { 0 }, { 0 }, { 0 }, { -1 }, 1, 1 } );

    Push_K_Stencil( p_state, K_Stencil{ 0, { 1, 0, 0 } } );
    Push_B_Stencil( p_state, B_Stencil{ 0, { 1, 0, 0 } } );

    float vec[3] = { 2, 2, 2 };
    Set_Domain( p_state, vec );

    State_Allocate( p_state );
    State_Upload( p_state );

    Set_Method( p_state, Method::Minimise, Solver::VP );
    Iterate( p_state, N_iterations );

    State_Delete( p_state );
}