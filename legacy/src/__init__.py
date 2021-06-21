"""
# Cartan Quantum Synthesizer

This code implements the algorithm(s) found in:

Fixed Depth Hamiltonian Simulation via Cartan Decomposition
Efekan Kökcü, Thomas Steckmann, J. K. Freericks, Eugene F. Dumitrescu, Alexander F. Kemper
https://arxiv.org/abs/2104.00728

_Abstract_:
Simulating spin systems on classical computers is challenging for large systems due to the significant memory requirements. This makes Hamiltonian simulation by quantum computers a promising option due to the direct representation of quantum states in terms of its qubits. Standard algorithms for time evolution on quantum computers require circuits whose depth grows with time. We present a new algorithm, based on Cartan decomposition of the Lie algebra generated by the Hamiltonian, that generates a circuit with time complexity O(1) for ordered and disordered models of n spins. We highlight our algorithm for special classes of models where an O(n^2)-gate circuit emerges. Compared to product formulas with significantly larger gate counts, our algorithm drastically improves simulation precision. Our algebraic technique sheds light on quantum algorithms and will reduce gate requirements for near term simulation.

Authors:
  Efekan Kökcü
  Thomas Steckmann

# Documentation
Please See the [Documation](docs/) for Information on the Functions included in the package. Documetation generated by pdoc in the google style. A google style reference document is included. 

To recompile the documentation, run 
`pdoc -d google -o .\docs .\src` 
from the main folder (requires pdoc (not pdoc3 or pdocs) 

# Current State:
Not currently functioning. Requires Code from Efekan implementing more complete methods

Examples:
    This is a test field for where examples would go
"""