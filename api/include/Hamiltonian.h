#pragma once
#ifndef API_HAMILTONIAN_H
#define API_HAMILTONIAN_H
#include "State.h"

struct ED_Stencil
{
    int i;
    int j[1];
    int da[1];
    int db[1];
    int dc[1];
    float J;
    float D;
};

struct K_Stencil
{
    int i;
    float K[3];
};

struct B_Stencil
{
    int i;
    float B[3];
};

PREFIX void Push_ED_Stencil( State * state, ED_Stencil stencil ) SUFFIX;
PREFIX void Set_ED_Stencils( State * state, ED_Stencil * ed_stencils, int N ) SUFFIX;

PREFIX void Push_K_Stencil( State * state, K_Stencil stencil ) SUFFIX;
PREFIX void Set_K_Stencils( State * state, K_Stencil * k_stencils, int N ) SUFFIX;

PREFIX void Push_B_Stencil( State * state, B_Stencil stencil ) SUFFIX;
PREFIX void Set_B_Stencils( State * state, B_Stencil * ed_stencils, int N ) SUFFIX;

#endif