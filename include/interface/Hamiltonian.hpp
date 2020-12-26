#pragma once
#ifndef HAMILTONIAN2_HPP
#define HAMILTONIAN2_HPP
#include "interface/Stencil.hpp"
#include <array>
#include <vector>

namespace Spirit
{
namespace Host
{

struct Hamiltonian
{
    std::array<bool, 3> boundary_conditions;
    std::vector<Spirit::Device::Stencil<2, Matrix3>> ed_stencils;
    std::vector<Spirit::Device::Stencil<1, Vector3>> k_stencils;
    std::vector<Spirit::Device::Stencil<1, Vector3>> b_stencils;
};

} // namespace Host
} // namespace Spirit

#endif