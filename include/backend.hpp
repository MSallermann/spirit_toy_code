#pragma once
#ifndef BACKEND_HPP
#define BACKEND_HPP
#include "State.hpp"
#include "definitions.hpp"

#ifdef DEBUG_PRINT
#include <iostream>
#include <string>
#endif

template<typename T>
inline void debug_print( T msg )
{
#ifdef DEBUG_PRINT
    std::cout << msg << "\n";
#endif
}

std::string description();
void create_backend_handle( State & state );
void iterate( Backend_Handle & state, int N_iterations );

#endif