#pragma once
#ifndef API_STATE_H
#define API_STATE_H
#include "DLL_Define_Export.h"

struct State;

PREFIX State * State_Setup( int n_cells[3], int n_cell_atoms, bool boundary_conditions[3] ) SUFFIX;
PREFIX void State_Delete( State * state ) SUFFIX;

PREFIX void State_Download( State * state ) SUFFIX;
PREFIX void State_Upload( State * state ) SUFFIX;
PREFIX void State_Allocate( State * state ) SUFFIX;

#endif