#include "interface/Method.hpp"
#include "implementation/Method_Minimize.hpp"
#include "implementation/Solver_Gradient_Descent.hpp"

namespace Spirit
{

Solver_Implementation * get_solver_implementation( Host::State * state, SolverType type )
{
    if( type == SolverType::Gradient_Descent )
    {
        return new Device::Solver_Gradient_Descent();
    }
    else
    {
        return nullptr;
    }
}

Method_Implementation * get_method_implementation( Host::State * state, MethodType type )
{
    if( type == Minimisation )
    {
        std::cout << "create minimize\n";
        return new Device::Method_Minimize( state );
    }
    else
    {
        return nullptr;
    }
}

} // namespace Spirit
