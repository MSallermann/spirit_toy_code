#include "interface/State.hpp"
#include "implementation/Fields.hpp"
#include "implementation/Hamiltonian.hpp"
#include <iostream>

namespace Spirit
{
namespace Interface
{

void State::allocate()
{
    if( this->fields != nullptr )
    {
        delete this->fields;
    }
    this->fields           = new Implementation::Fields();
    this->fields->spins    = Implementation::device_vector<Vector3>( this->geometry.nos );
    this->fields->gradient = Implementation::device_vector<Vector3>( this->geometry.nos );
}

void State::upload()
{
    this->fields->spins.copy_from( this->spins );

    if( this->hamiltonian_device != nullptr )
    {
        delete this->hamiltonian_device;
    }
    this->hamiltonian_device = new Implementation::Hamiltonian( &( this->hamiltonian ) );
}

void State::download()
{
    this->fields->spins.copy_to( this->spins );
    this->fields->gradient.copy_to( this->gradient );
}

State::~State()
{
    if( this->fields != nullptr )
    {
        delete this->fields;
    }
    if( this->hamiltonian_device != nullptr )
    {
        delete this->hamiltonian_device;
    }
}

} // namespace Interface
} // namespace Spirit
