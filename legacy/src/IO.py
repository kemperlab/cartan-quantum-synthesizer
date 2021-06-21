# -*- coding: utf-8 -*-
__docformat__ = "google"
"""
Created on Mon Dec 21 15:58:24 2020

@author: Thomas Steckmann
@author: Efekn Kokcu
"""
from numpy import kron
from src.PauliOps import X,Y,Z,I,paulis


def printlist(tuples):
    """Function to Print from Tuple format to string format Pauli Strings

    Args:
        tuples (List of Tuples): List of Pauli Strings to Convert to text
    
    """
    res = ''
    
    chars = 0
    for p in tuples:
        for i in range(len(p)):
            if p[i] == 0:
                res = res + 'I'
            elif p[i] == 1:
                res = res + 'X'
            elif p[i] == 2:
                res = res + 'Y'
            elif p[i] == 3:
                res = res + 'Z'
            chars = chars + 1
        
        res = res + ', '
        
        if chars > 70:
            res = res + '\n'
            chars = 0
        
    
    print(res+'\n\n\n')
    
    

def tuplesToMatrix(coefficient, PauliTuple):
    """
    Converts a Pauli String represented as a Tuple to a matrix element that can be operated on traditionally. Generally expensive to operate with

    Args:
        coefficient (np.Complex128):
            Multiplies the resultant matrix
        PauliTuple (Tuple)
            Pauli Tuple of the form (0, 1, 2, 3) == IXYZ == kron(kron(kron(I, X), Y), Z)
    Returns:
        ndarray, equal to the kronecker product of the Pauli elements, multiplied by the coefficients
"""
    if len(PauliTuple) == 1:
        #paulis is a list of the pauli matricies, so we take the index of paulis corresponding to the Pauli Matrix in the Tuple
        return paulis[PauliTuple[0]] * coefficient
    else:
        #We start with the kronecker product of the first to elements
        result = kron(paulis[PauliTuple[0]], paulis[PauliTuple[1]])
        for i in range(2,len(PauliTuple)):
            result = kron(result,paulis[PauliTuple[i]])
        return result * coefficient



"""
Used to Verify functions in this module
"""

'''
import scipy.linalg as la
import numpy as np
coefficient = 1 + 3j

x = kron(kron(kron(X,Y),Z),I)
xT = (1,2,3,0)
print(la.norm(tuplesToMatrix(coefficient,xT) - x* coefficient))
'''
        
    