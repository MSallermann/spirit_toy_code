#pragma once
#ifndef IMPLEMENTATION_STENCIL_EVALUATOR_HPP
#define IMPLEMENTATION_STENCIL_EVALUATOR_HPP

#ifdef BACKEND_CPU
#include "implementation/Stencil_Evaluator.cpu.hpp"
#endif

#ifdef BACKEND_CUDA
#include "implementation/Stencil_Evaluator.cuda.hpp"
#endif

#endif