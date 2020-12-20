#ifdef BACKEND_VULKAN
#include "backend.hpp"

std::string description()
{
    std::string des;
    des = "VULKAN";
    return des;
}

void gradient() {}

void iterate( State & state, int N_iterations ) {}

#endif