#pragma once
#ifndef BACKEND_HANDLE_HPP
#define BACKEND_HANDLE_HPP
#include "definitions.hpp"
#include <memory>

class State;

struct Backend_Handle
{
    Backend_Handle(State & state);
    ~Backend_Handle();

    void Upload( State & state );
    void Download( State & state );

    // The *device* pointer to this Backend Handle
    Backend_Handle * dev_ptr;

    int * n_cells;
    int n_cell_atoms;
    Vector3 * spins;
    Pair_Stencil * pair_stencils;
    int N_pair;
    int nos;
    scalar timestep;
    Vector3 * gradient = nullptr;
    // State * state;
};

#endif