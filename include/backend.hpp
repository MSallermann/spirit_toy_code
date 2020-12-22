#pragma once
#ifndef BACKEND_HPP
#define BACKEND_HPP
#include "Host_State.hpp"
#include "definitions.hpp"
namespace Spirit
{
namespace Device
{

std::string description();
void iterate( Host_State & state, int N_iterations );

} // namespace Device
} // namespace Spirit
#endif