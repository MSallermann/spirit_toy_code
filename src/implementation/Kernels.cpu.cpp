#ifdef BACKEND_CPU
#include "implementation/Kernels.hpp"
#include "implementation/Lambda.hpp"

namespace Spirit
{
namespace Implementation
{
namespace Kernels
{

void set_gradient_zero( Vector3 * gradient, Interface::State::Geometry & geometry )
{
#pragma omp parallel for
    for( int i = 0; i < geometry.nos; i++ )
    {
        gradient[i] = { 0, 0, 0 };
    }
}

void propagate_spins(
    Vector3 * spins, Vector3 * gradient, Interface::State::Solver_Parameters & solver_parameters, Interface::State::Geometry & geometry )
{
#pragma omp parallel for
    for( int idx = 0; idx < geometry.nos; idx++ )
    {
        spins[idx] += solver_parameters.timestep * gradient[idx];
        spins[idx].normalize();
    }
}

Summator::Summator( int N ) : N( N ){};
Summator::~Summator(){};
void Summator::sum( scalar * result, scalar * scalarfield )
{
    scalar res = 0;
#pragma omp parallel for reduction( + : res )
    for( int idx = 0; idx < N; idx++ )
    {
        res += scalarfield[idx];
    }
    *result     = res;
    last_result = result;
};

scalar Summator::download_last_result()
{
    return *last_result;
}

void dot( scalar * result, Vector3 * vf1, Vector3 * vf2, int N )
{
    Lambda::apply( N, [result, vf1, vf2] SPIRIT_LAMBDA( int idx ) { result[idx] = vf1[idx].dot( vf2[idx] ); } );
}

} // namespace Kernels
} // namespace Implementation
} // namespace Spirit
#endif
