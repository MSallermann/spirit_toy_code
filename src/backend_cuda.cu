#ifdef BACKEND_CUDA
#include "backend.hpp"

void create_backend_handle(State & state)
{
    Backend_Handle& res = state.backend;
    res.n_cells         = state.n_cells.data();
    res.n_cell_atoms    = state.n_cell_atoms;
    res.spins           = state.spins.data();
    res.pair_stencils   = state.pair_stencils.data();
    res.N_pair          = state.pair_stencils.size();
    res.timestep        = state.timestep;
    res.nos             = state.Nos();
    res.gradient        = std::vector<Vector3>(res.nos);
}

void gradient(Backend_Handle & state)
{
    int Na = state.n_cells[0];
    int Nb = state.n_cells[1];
    int Nc = state.n_cells[2];
    int idx_i, idx_j;

    for(Vector3 & g : state.gradient)
    {
        g = {0,0,0};
    }

    for(int c=0; c<Nc; c++)
    {
        for(int b=0; b<Nb; b++)
        {
            for(int a=0; a<Na; a++)
            {
                for(int i=0; i<state.n_cell_atoms; i++)
                {
                    idx_i = i + state.n_cell_atoms * (a + Na * (b + Nb * c));
                    for(int p=0; p<state.N_pair; p++)
                    {
                        const Pair_Stencil & pair = state.pair_stencils[p];
                        idx_j = pair.j + state.n_cell_atoms * ( (a + pair.da) + Na * (b + pair.db + Nc * (c + pair.dc) ) );
                        if(i==pair.i && idx_j>0 && idx_j<state.nos)
                        {
                            state.gradient[idx_i] += pair.matrix * state.spins[idx_j];
                        }
                    }
                }
            }
        }
    }
}

void propagate_spins(Backend_Handle & state)
{
    for(int idx=0; idx<state.nos; idx++)
    {
        state.spins[idx] += state.timestep * state.gradient[idx];
        state.spins[idx].normalize();
    }
}

void iterate(Backend_Handle & state, int N_iterations)
{
    for(int iter=0; iter<N_iterations; iter++)
    {
        gradient(state);
        propagate_spins(state);
    }
}

#endif