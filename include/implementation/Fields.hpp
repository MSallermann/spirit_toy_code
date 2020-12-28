#pragma once
#ifndef IMPLEMENTATION_STATE_HPP
#define IMPLEMENTATION_STATE_HPP
#include "Definitions.hpp"
#include "implementation/Memory.hpp"

namespace Spirit
{
namespace Implementation
{

class Fields
{
public:
    device_vector<Vector3> spins;
    device_vector<Vector3> gradient;
};

} // namespace Implementation
} // namespace Spirit

#endif