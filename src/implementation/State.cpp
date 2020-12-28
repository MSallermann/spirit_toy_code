#include "implementation/State.hpp"
#include "implementation/Stencil_Evaluator.hpp"

namespace Spirit
{
namespace Implementation
{

void State::get_gradient( Vector3 * gradient, Vector3 * spins, State_Pod & state_pod )
{
    if( hamiltonian.ed_stencils.size() > 0 )
    {
        stencil_gradient( gradient, spins, state_pod, hamiltonian.ed_stencils.size(), hamiltonian.ed_stencils.data() );
    }
    if( hamiltonian.k_stencils.size() > 0 )
    {
        stencil_gradient( gradient, spins, state_pod, hamiltonian.k_stencils.size(), hamiltonian.k_stencils.data() );
    }
    if( hamiltonian.b_stencils.size() > 0 )
    {
        stencil_gradient( gradient, spins, state_pod, hamiltonian.b_stencils.size(), hamiltonian.b_stencils.data() );
    }
}

} // namespace Implementation
} // namespace Spirit