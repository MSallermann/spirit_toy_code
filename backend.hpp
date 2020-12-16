#pragma once
#ifndef BACKEND_HPP
#define BACKEND_HPP
#include "State.hpp"
#include "definitions.hpp"

void create_backend_handle(State & state);
void iterate(Backend_Handle & state, int N_iterations);

#endif