#include "interface/Method.hpp"
#include "implementation/Method_Minimize.hpp"
#include "implementation/Solver_Gradient_Descent.hpp"
#include "implementation/Solver_VP.hpp"

namespace Spirit
{
namespace Interface
{

Method::Method( Interface::State & state, MethodType type )
{
    if( type == Minimisation )
    {
        this->m_implementation = new Implementation::Method_Minimize( state );
    }
    else
    {
        this->m_implementation = nullptr;
    }
}

void Method::set_solver( SolverType type )
{
    if( type == SolverType::Gradient_Descent )
    {
        this->m_implementation->set_solver( new Implementation::Solver_Gradient_Descent( m_implementation->state ) );
    }
    else if( type == SolverType::VP )
    {
        this->m_implementation->set_solver( new Implementation::Solver_VP( m_implementation->state ) );
    }
}

} // namespace Interface
} // namespace Spirit