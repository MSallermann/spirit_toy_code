#pragma once
#ifndef BACKEND_HPP
#define BACKEND_HPP
#include "Host_State.hpp"
#include "definitions.hpp"

std::string description();
void iterate( Host_State & state, int N_iterations );

#endif