import random

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Pauli
from qiskit.transpiler.passes.synthesis.plugin import HighLevelSynthesisPlugin

import CQS.util.IO as IO
from CQS.methods import Cartan, FindParameters, Hamiltonian

# To check transpiled circuit has same matrix norm, convert both original and transpiled
# qcs to operators then use the equiv() method from the quantum_info module.
# For basis gates, use RX, RY, RZ, CX.
# translation_method choice (unroller vs. translator) shouldn't matter too much.
# For Hamiltonian, use the 1D Heisenberg model with Next-to nearest and nearest neighbour
# coupling with open boundary conditions (NNN & open).

# 3q NNN-open on 3 linear qubits with: IXX, XXI, ...
# 3q NNN-open on 4 Y-shaped qubits: IIXX, IXXI, ...
# i.e. same Hamiltonian except append an identity for the leftmost qubit,
# with coupling_map = [[0, 1], [0, 2], [0, 3], [1, 0], [2, 0], [3, 0]],
# i.e qubit 0 is the central qubit.


# Define the function to create a Qiskit QuantumCircuit given Cartan parameters.
def generate_cartan_circuit(CQS_Cartan, time_evolve):
    # Generate the parameters via classical optimization of the cost function.
    CQS_parameters = FindParameters(CQS_Cartan)

    # Store optimization results.
    kTuples = CQS_parameters.cartan.k
    kCoefs = CQS_parameters.kCoefs
    hTuples = CQS_parameters.cartan.h
    hCoefs = CQS_parameters.hCoefs

    # Retrieve number of qubits from Cartan object.
    num_qubits = CQS_Cartan.hamiltonian.sites

    # Generate Qiskit Quantum Circuit that implemenets the Cartan Decomp.
    qc = QuantumCircuit(num_qubits)

    # K^\dag.
    for kTuple, kCoef in zip(kTuples, kCoefs):
        kString = str(IO.paulilabel(kTuple))
        gate = PauliEvolutionGate(Pauli(kString), time=kCoef)
        qc.append(gate, range(num_qubits))

    qc.barrier()

    # h.
    for hTuple, hCoef in zip(hTuples, hCoefs):
        hString = str(IO.paulilabel(hTuple))
        gate = PauliEvolutionGate(
            Pauli(hString), time=np.real(hCoef) * time_evolve
        )  # WLOG convert complex to real
        qc.append(gate, range(num_qubits))

    qc.barrier()

    # K.
    for kTuple, kCoef in reversed(list(zip(kTuples, kCoefs))):
        kString = str(IO.paulilabel(kTuple))
        gate = PauliEvolutionGate(Pauli(kString), time=-kCoef)
        qc.append(gate, range(num_qubits))

    return qc


# Define function to convert each Pauli string to a tuple,
# e.g "IXYZ" -> (0, 1, 2, 3).
def pauli_string_to_tuple(pauli_string):
    pauli_mapping = {"I": 0, "X": 1, "Y": 2, "Z": 3}
    pauli_tuple = tuple(pauli_mapping[pauli] for pauli in pauli_string)

    return pauli_tuple


# Define the function to synthesize a given PauliEvolutionGate.
def synth_cartan(paulievolutiongate, random_seed, involution="evenOdd"):
    """Cartan synthesis of a PauliEvolutionGate instance based on the method developed by the Kemper group.

    Args:
        paulievolutiongate (PauliEvolutionGate): a high-level definition of the unitary which implements
                                                 the time evolution under a Hamiltonian consisting of Pauli terms.
        random_seed (Int): seed to set the ordering of factors in K and the starting element of h.
                           Avoid using 0 to prevent unusual behavior.
                           If equal to -1, use the lexicographic ordering of pauli factors in K.
                           Otherwise, use a random ordering.
        involution (Str): The involution used for the Cartan decomposition.
                            The default involution is "evenOdd". Other options include
                            "knejaGlaser", "countX", "countY", "countZ".

    Return:
        QuantumCircuit: a circuit implementation of the input PauliEvolutionGate via a Cartan decomposition.

    Raises:
        QiskitError: if arg is not an instance of PauliEvolutionGate.
    """

    # Raise an error if the object to be synthesized is not a PauliEvolution.
    if not isinstance(paulievolutiongate, PauliEvolutionGate):
        raise TypeError(
            "The Cartan synthesis plugin can only synthesize PauliEvolution instances, "
            f"but it encountered an operation of type {type(paulievolutiongate)}."
        )

    # Get the number of qubits of the PauliEvolution.
    num_qubits = paulievolutiongate.num_qubits

    # Get the time to evolve for.
    time_evolve = paulievolutiongate.time

    # Get the Hamiltonian of the PauliEvolution.
    Ham = paulievolutiongate.operator

    # Get each PauliString and corresponding coefficient.
    Ham_Paulis = Ham.paulis
    Ham_PauliStrings = [pauli.to_label() for pauli in Ham_Paulis]

    # Convert each string to a tuple.
    Ham_PauliTuples = list(pauli_string_to_tuple(string) for string in Ham_PauliStrings)
    Ham_PauliCoeffs = list(Ham.coeffs)

    # Make tuple of lists to later create Hamiltonian object.
    Ham_terms = tuple([Ham_PauliCoeffs, Ham_PauliTuples])

    # Make empty CQS Hamiltonian object.
    CQS_Ham = Hamiltonian(num_qubits)

    # Add each term to CQS Hamiltonian.
    CQS_Ham.addTerms(Ham_terms)

    # Check each term has been added.
    # print(CQS_Ham.getHamiltonian(type="printText"))

    # Try to perform a Cartan involution on the Hamiltonian
    # using the defauls evenOdd Decomposition.
    # If H is not contained in the -1 eigenspace, an error is raised.
    CQS_Cartan = Cartan(CQS_Ham, manualMode=1)
    CQS_Cartan.g = CQS_Cartan.makeGroup(CQS_Cartan.HTuples)
    CQS_Cartan.decompose(involutionName=involution)

    # Set the chosen random seed.
    random.seed(random_seed)

    # Sort factors of k lexicographically in placeif random_seed == -1;
    # otherwise, randomly shuffle elements of k in place.
    if random_seed == -1:
        CQS_Cartan.k.sort()
    else:
        random.shuffle(CQS_Cartan.k)

    # Randomly choose an element of m as a starting element
    # of the Cartan subalgebra h.
    CQS_Cartan.subAlgebra(seedList=[random.choice(CQS_Cartan.m)])

    # Optimize cost function and create circuit.
    qc = generate_cartan_circuit(CQS_Cartan, time_evolve)

    return qc


# Define the High Level Synthesis Plugin.
class CartanPlugin(HighLevelSynthesisPlugin):
    def run(self, PauliEvolution, random_seed, involution, **kwarg):
        print("Running Cartan Synthesis Plugin...")

        return synth_cartan(PauliEvolution, random_seed, involution)
