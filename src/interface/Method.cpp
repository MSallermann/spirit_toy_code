#include "interface/Method.hpp"
#include "implementation/Method_Minimize.hpp"
#include "implementation/Solver_Gradient_Descent.hpp"
#include "implementation/Solver_VP.hpp"

namespace Spirit
{
namespace Interface
{

Solver_Implementation * get_solver_implementation( Interface::State & state, SolverType type )
{
    if( type == SolverType::Gradient_Descent )
    {
        return new Implementation::Solver_Gradient_Descent( state );
    }
    else if( type == SolverType::VP )
    {
        return new Implementation::Solver_VP( state );
    }
    else
    {
        return nullptr;
    }
}

Method_Implementation * get_method_implementation( Interface::State & state, MethodType type )
{
    if( type == Minimisation )
    {
        return new Implementation::Method_Minimize( state );
    }
    else
    {
        return nullptr;
    }
}

} // namespace Interface
} // namespace Spirit