#pragma once
#ifndef BACKEND_HANDLE_HPP
#define BACKEND_HANDLE_HPP
#include "definitions.hpp"
#include <memory>

class State;

struct Backend_Handle
{
    Backend_Handle();
    ~Backend_Handle();

    void Allocate( State & state ); // Allocates buffers for spins, gradient, pairs and updates them with the host information
    void Upload( State & state );
    void Download( State & state );

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