#ifdef BACKEND_VULKAN
#include "backend.hpp"
namespace Spirit
{
namespace Device
{

std::string description()
{
    std::string des;
    des = "VULKAN";
    return des;
}

void gradient() {}

void iterate( State & state, int N_iterations ) {}

} // namespace Device
} // namespace Spirit
#endif