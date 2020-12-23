# Notes
Documenting my thinking process ...

# Ideas
- Should make the separation between host and device reflected in namespaces
- Maybe one can abstract the methods and solvers into a series of "actions" wich would be started from the host and could be accelerated according to the used backend. This would make it possible to "compose" the methods form different actions, which are defined in the backend interface. One action could be computing the energy gradient for example. 

I made three namespaces: `Spirit`, `Spirit::Host` and `Spirit::Device`
- In the `Spirit` namespace the logic of the methods and solvers should be implemented (using the abstract)
- In the `Spirit::Device` namespace the implementation of the "actions" is
- In the `Spirit::Host` namespace 

# Things to still be implemented
- [ ] energy calculation in minimizers
- [ ] multiple solvers (maybe gradient descent and VP for a start)
- [ ] multiple methods (maybe energy minimisation and LLG for a start)
- [ ] dummy API with getters setters
- [ ] Vulkan Backend

# Design rules I thought of (for now)
- `Device_State` should be able to describe itself fully to every backend function -> signature of all backend functions becomes `T function(Device_State * state)`
- `Device_State` should have absolutely no knowledge of the Host_State

# How to implement a new stencil interaction (for now)
1. In `Hamiltonian.hpp`: Declare a new Stencil class, which is derived from the appropriately templated `Stencil<N,PARAM>` base class 
2. In `Host_State.hpp`: 
    - Add a new class member `std::vector<New_Stencil> new_stencils`
    - Add initialization of new class members to the constructor (optionally implement getters and setters)
3. In `Device_State.hpp`: 
    - Add a pointer `New_Stencil * new_stencils` to the members
    - Add counter `n_new_stencil` to the members
    - Add template specializations of `get_n_stencil<New_Stencil>` and `__get_stencil<New_Stencil>` *below* the class definition
4. In `Device_State.cpp/cu/xxx`: 
    - Add `malloc_n` call to `Device_State::allocate` (malloc to `new_stencils`)
    - Copy `state.new_stencils.size()` to `n_new_stencil`
    - Add `copy_vector_H2D` call to `Device_State::upload`
5. Hook up the call to `stencil_gradient<N, New_Stencil>` in `Backend::iterate`

# How to implement a new method
To be added ...

# How to implement a new solver
To be added ...