#pragma once
#ifndef API_METHOD_H
#define API_METHOD_H
#include "State.h"

enum Method
{
    Minimise = 0,
    LLG      = 1,
    None     = 2
};

enum Solver
{
    Gradient_Descent = 0,
    VP               = 1
};

PREFIX void Set_Method( State * state, Method type, Solver solver ) SUFFIX;
PREFIX void Iterate( State * state, int N_iterations ) SUFFIX;

#endif