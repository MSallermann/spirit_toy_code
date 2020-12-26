#pragma once
#ifndef BACKEND_HPP
#define BACKEND_HPP
#include "definitions.hpp"
#include "implementation/Host_State.hpp"
namespace Spirit
{
namespace Device
{

std::string description();
void iterate( Spirit::Host::Host_State & state, int N_iterations );

} // namespace Device
} // namespace Spirit
#endif