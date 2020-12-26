#include "interface/State.hpp"
#include "implementation/State.hpp"
#include <iostream>

namespace Spirit
{
namespace Host
{

void State::allocate()
{
    if( this->device_state != nullptr )
    {
        delete this->device_state;
    }
    this->device_state = new Spirit::Device::State( this );
}

void State::upload()
{
    this->device_state->spins.copy_from( this->spins );
}

void State::download()
{
    this->device_state->spins.copy_to( this->spins );
    this->device_state->gradient.copy_to( this->gradient );
}

State::~State()
{
    if( this->device_state != nullptr )
    {
        delete this->device_state;
    }
}

} // namespace Host
} // namespace Spirit
