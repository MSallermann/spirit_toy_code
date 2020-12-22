#pragma once
#ifndef STATE_HPP
#define STATE_HPP

#include "Device_State.hpp"
#include "Host_State.hpp"

namespace Spirit
{

class State
{
    Host::Host_State host_state;
    Device::Device_State device_state;

    State( Host::Host_State host_state, Device::Device_State device_state ) : host_state( host_state ), device_state( device_state ){};
};

} // namespace Spirit

#endif