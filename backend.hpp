#pragma once
#ifndef BACKEND_HPP
#define BACKEND_HPP
#include "State.hpp"
#include "definitions.hpp"

std::string description();
void iterate( State & state, int N_iterations );

#endif