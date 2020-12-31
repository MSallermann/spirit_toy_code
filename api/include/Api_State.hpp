#pragma once
#ifndef API_STATE_HPP
#define API_STATE_HPP
#include "interface/Method.hpp"
#include "interface/State.hpp"

struct State
{
public:
    State( int n_cells[3], int n_cell_atoms, bool boundary_conditions[3] )
            : core_state( { n_cells[0], n_cells[1], n_cells[2] }, n_cell_atoms, { true, true, true } ),
              method( core_state, Spirit::Interface::MethodType::None )

    {
    }
    Spirit::Interface::State core_state;
    Spirit::Interface::Method method;
};

#endif