#import sys
#sys.path.append('c:\\Users\\Thoma\\OneDrive\\Documents\\2021_ORNL\\CartanCodeGit\\cartan-quantum-synthesizer')
# -*- coding: utf-8 -*-
__docformat__ = 'google'
"""
A collection of functions useful for exact diagonalization and converting KHK decomposition to a matrix
"""

import numpy as np
from numpy import kron
from scipy.linalg import expm, norm

import CQS.util.IO as IO

#The Pauli Matricies in matrix form
X = np.array([[0,1],[1,0]])
#Pauli X
Y = np.array([[0,-1j],[1j,0]])
#Pauli Y
Z = np.array([[1,0],[0,-1]])
#PauliZ
I = np.array([[1,0],[0,1]])
#2x2 idenity

paulis = [I,X,Y,Z]
# Allows for indexing the Pauli Arrays (Converting from tuple form (0,1,2,3) to string form IXYZ)

def Nident (N):
    """ Generates an N qubit Identity Matrix """

    return np.diag(np.ones(2**N))


def PauliExpUnitary(N, co, PauliTuple):
    """
    Generates the Unitary Matrix for a Pauli Exponential
    Uses e^{i.co.Pauli} = I*cos(a) + i*sin(a)*Pauli

    Args:
        N (int): Number of qubits
        co (float): The coefficient of the Pauli Matrix
        PauliTuple (Tuple): (PauliString) to exp
    
    Returns:
        The result e<sup>i•co•PauliTuple</sup> = I•cos(co) + i•sin(co)•PauliTuple
    """
    II = Nident(N)

    U = paulis[PauliTuple[0]]

    for pauli in PauliTuple[1:]:
        U = kron(U,paulis[pauli]) #Generates the PauliTuple Matrix Element
    return np.cos(co)*II + 1j*np.sin(co)*U


def exactU(HCos, HTups, time):
    """
    Computes the exact matrix exponential for time evolution at the time t. Takes as an input the real component of the exponential.
    
    Args:
        HCos (List of complex numbers):
        HTupes (List of (PauliStrings)):
        time (float - time evolution final time):


    """
    H = np.diag(np.zeros(2**len(HTups[0])))
    for (co, term) in zip(HCos, HTups):
        H = H + IO.tuplesToMatrix(co, term)
    return expm(1j * time * H)



def Trotter(ham, time, N, steps):
    """
    Prepares U_t, the Trotterized input U

    Args:
        ham (List of Tuples): Hamiltonian formatted as (co, (PauliString))
        time (float): final time to evolve to
        N (int): number of qubits
        steps (int): Number of trotter steps to take
    
    Returns:
        The U<sub>trotter</sub>(t) that approximates U<sub>exact</sub>(t)
    """
    timeStep = time/steps
    U = Nident(N)
    for (co, pauliTuple) in ham:
        U = U @ PauliExpUnitary(N, 1*co*timeStep, pauliTuple)
    finalU = Nident(N)
    for i in range(steps):
        finalU = finalU @ U
        
    return finalU


def KHK(kCoefs, hCoefs, k, h):
    """
    Defines the Unitary for the KHK<sup>†</sup>]

    Specifically, performs ℿ<sub>i</sub> e<sup>i•k[l]•kCoefs[l]</sup> ℿ<sub>j</sub> e<sup>i•h[j]•hCoefs[j]</sup>  ℿ<sub>l</sub> e<sup>i•k[(lenK - l)]•kCoefs[(lenK - l)]</sup> 

    Multiply by t before passing the coefficients for h. Do not multiply h by i, that is automatic. The coefficients should be real for k, imaginary for h

    Args:
        kCoefs (List): A list of (real) coefficients for k
        hCoefs (List): The list of (imaginary) coefficients for the elements in h. 
        k (List of Tuples): The list of (PauliStrings)
        h (List of Tuples): List of (PauliStrings) for h (in the same indexing)


    """
    N = len(h[0])
    KHK = Nident(N)
    
    #First loop of K terms:
    for (term, co)  in zip(k, kCoefs):
        KHK = KHK @ PauliExpUnitary(N, co, term)
    #H terms
    for (term, co) in zip(h, hCoefs):
        KHK = KHK @ PauliExpUnitary(N, co, term)
    for (term, co)  in zip(k[::-1], kCoefs[::-1]):
        KHK = KHK @ PauliExpUnitary(N, -1*co, term)
    return KHK