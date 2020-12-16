#ifndef STATE_HPP
#define STATE_HPP
#include "definitions.hpp"
#include <array>
#include <vector>

struct Pair_Stencil
{
    int i;
    int j;
    int da;
    int db;
    int dc;
    Matrix3 matrix;
};

struct Backend_Handle
{
    int* n_cells; 
    int n_cell_atoms; 
    Vector3* spins;
    Pair_Stencil* pair_stencils;
    int N_pair;
    int nos;
    scalar timestep;
    std::vector<Vector3> gradient;
};

class State
{
    public:
    State(std::array<int, 3> n_cells, int n_cell_atoms, scalar timestep);
    void Set_Pair_Stencils(std::vector<Pair_Stencil> & stencils);
    void Create_Backend_Handle();

    int Nos()
    {
        return n_cell_atoms * n_cells[0] * n_cells[1] * n_cells[2];
    }
    
    Backend_Handle backend;
    std::array<int, 3> n_cells; 
    int n_cell_atoms; 
    scalar timestep;
    std::vector<Vector3> spins;
    std::vector<Pair_Stencil> pair_stencils;

};

#endif