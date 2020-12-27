#include "implementation/State.hpp"
#include "implementation/Stencil_Evaluator.hpp"

namespace Spirit
{
namespace Device
{

void State::Gradient_Async( device_vector<Vector3> & gradient, State & state )
{
    if( hamiltonian.ed_stencils.size() > 0 )
    {
        stencil_gradient( state, hamiltonian.ed_stencils.size(), hamiltonian.ed_stencils.data() );
    }
    if( hamiltonian.k_stencils.size() > 0 )
    {
        stencil_gradient( state, hamiltonian.k_stencils.size(), hamiltonian.k_stencils.data() );
    }
    if( hamiltonian.b_stencils.size() > 0 )
    {
        stencil_gradient( state, hamiltonian.b_stencils.size(), hamiltonian.b_stencils.data() );
    }
}

} // namespace Device
} // namespace Spirit