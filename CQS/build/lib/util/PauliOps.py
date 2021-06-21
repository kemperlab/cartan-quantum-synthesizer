# -*- coding: utf-8 -*-
__docformat__ = 'google'
"""
Created on Mon Dec 21 15:16:56 2020
A Collection of Methods to calculate useful operations on Pauli Strings. Mostly commutators

@author: Thomas Steckmann
@author: Efekan Kokcu

"""
import numpy as np



ops = ['I','X','Y','Z']
"""Indices for converting from (PauliString) --> String format"""
RULES = [1,3,1,3]
"""
Used for generating the Commutator tables and pauli commutators (efficiently we think)
``
RULES:
    Used to find the multiplication between two paulis represented as indices in a tuple (I == 0, X == 1, Y == 2, Z == 3)
The operation is (index1 + index2*RULES[index1] % 4) = Pauli Matrix result as an index

I * anything: 0 + (Index2)*1 = index2
X * anythong: (1 + (Index2)*3 % 4) gives
                                         1 + 0 = 1 for I, 
                                         (1 + 1*3) % 4 = 0 for X
                                         (1 + 2*3) % 4 = 7 % 4 = 3 for Y
                                         (1 + 3*3) % 4 = 10 % 4 = 2 for Z as index2
These can easily be expanded for Y and Z
``
"""

SIGN_RULES = [[1,1,1,1], 
             [1, 1, 1j, -1j],
             [1, -1j, 1, 1j],
             [1, 1j, -1j, 1]]
"""
Rules for computing the sign of two commutators
```
SIGN_RULES: 
    Gives the multiplication sign rules for multiplying Pauli Matricies (ex. X*Y -> iZ)
    
  I  X  Y  Z
I +  +  +  +
X +  +  +i -i
Y +  -i +  +i
Z +  +i -i +

Order: row * column
```
"""

#The Pauli Matricies in matrix form
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
I = np.array([[1,0],[0,1]])
paulis = [I,X,Y,Z]
#Allows for indexing the Pauli Arrays (Converting from tuple form (0,1,2,3) to string form IXYZ)

def commutatePauliString(a,tupleA,b,tupleB, comm_coefs = None, comm_table = None):
    """Computes the commutator of two Pauli Strings representated as a tuple
    
    If a commutator table is passed, the operation is much more efficient

    Args:
        a (np.complex128): 
            The coefficient of the first Pauli String term
        tupleA (Tuple, integer): 
            tuple represenation of the first Pauli String, or the index in the commutator table
        b (np.complex128): 
            The coefficient of the second Pauli String term
        tupleB (tuple, int): 
            tuple represenation of the second Pauli String, or the index in the commutator table
        
    
    Returns:
        c (np.complex128):
            The coefficient of the result [a*TupleA,b*TupleB] = c*TupleC, where c is the Structure Constant * a * b
        tupleC (tuple): 
            the elementwise commutator of the PauliString, ignoring coefficients. 
    """
    
    if type(tupleA)==tuple:
    
        sites = len(tupleA)
        #builds up the result Tuple C
        tupleC = ()

        signForward = 1 # sign(a.b)
        signBackward = 1 # sign(b.a) 


        #Iterate elementwise over the tuple
        for i in range(sites):
            #tupleC is the tuple represenation of the result of the commutator 
            tupleC += (((tupleA[i] + tupleB[i]*RULES[tupleA[i]]) % 4),)
            #Complex integer product of all the elementwise multiplications going forward
            signForward = signForward * SIGN_RULES[tupleA[i]][tupleB[i]]
            #Complex integer product of all the elementwise multiplications going backward
            signBackward = signBackward * SIGN_RULES[tupleB[i]][tupleA[i]]

        #Checks the signs forward and backwards. If they are the same, it commutes
        if signForward == signBackward:
            return (0,tupleC)
        else:
            c = a * b * 2*signForward
            return c, tupleC
    
    else:
        return (a*b*comm_coefs[int(tupleA)][int(tupleB)]), comm_table[int(tupleA)][int(tupleB)] 
    
    
    

def multiplyPauliString(a,tupleA,b,tupleB):
    """Computes the multiplication of two Pauli Strings representated as a tuple

    Args:
        a (np.complex128): 
            The coefficient of the first Pauli String term
        tupleA (Tuple):
            tuple represenation of the first Pauli String
        b (np.complex128):
            The coefficient of the second Pauli String term
        tupleB (Tuple): 
            tuple represenation of the second Pauli String

    Returns:
        c (np.complex128):
            The coefficient of the result a*TupleA . b*TupleB = c*TupleC, where c (the sign of the product of Paulis * a * b)
        tupleC (tuple) :
            the elementwise product of the PauliString, ignoring coefficients. 
    """
      
    sites = len(tupleA)
    #builds up the result Tuple C
    tupleC = ()
    sign = 1 # sign(a.b)
    
    #Iterate elementwise over the tuple
    for i in range(sites):
        #tupleC is the tuple represenation of the result of the commutator 
        tupleC += (((tupleA[i] + tupleB[i]*RULES[tupleA[i]]) % 4),)
        #Complex integer product of all the elementwise multiplications
        sign = sign * SIGN_RULES[tupleA[i]][tupleB[i]]
        
    c = a * b * sign
    return c, tupleC  



def multiplyLinComb(A,tuplesA,B,tuplesB):
    '''Returns multiplication of two linear combinations of Pauli terms 
    '''

    a = len(A)
    b = len(B)
    
    C = []
    tuplesC = []
    csize = 0
    
    for i in range(a):
        for j in range(b):
            term = multiplyPauliString(A[i],tuplesA[i],B[j],tuplesB[j])
            flag = 0
            for k in range(csize):
                if tuplesC[k]==term[1]:
                    flag = 1
                    C[k] = C[k]+term[0]
            if flag == 0:
                C.append(term[0])
                tuplesC.append(term[1])
                csize = csize + 1
    
    return C, tuplesC





def simplifyLinComb(A,tuples):
    '''Modifies the input lists
    
    Simplifies lin comb of Pauli matrices that it eats. Doens't return anything
    Args:
        A: A list
        tuples: A list
    '''
    
    size = len(A)
    
    index = 0
    
    while index < size:
        flag = 0
        for i in range(index):
            if tuples[i]==tuples[index]:
                A[i] = A[i]+A[index]
                A.pop(index)
                tuples.pop(index)
                flag = 1
                size = size-1
                break
                
        if flag == 0:
            index = index + 1
            
def commutateLinComb(A,tuplesA,B,tuplesB,accur):
    a = len(A)
    b = len(B)
    
    C = []
    tuplesC = []
    csize = 0
    
    for i in range(a):
        for j in range(b):
            term = commutatePauliString(A[i],tuplesA[i],B[j],tuplesB[j])
            flag = 0
            for k in range(csize):
                if tuplesC[k]==term[1]:
                    flag = 1
                    C[k] = C[k]+term[0]
            if (flag == 0) & (abs(term[0])>accur):
                C.append(term[0])
                tuplesC.append(term[1])
                csize = csize + 1
    
    return C, tuplesC



def multiplyLinCombRound(A,tuplesA,B,tuplesB, accur):
    '''
    Returns multiplication of two linear combinations of Pauli terms, and rounds things that are smaller than accur to zero. 
    '''

    a = len(A)
    b = len(B)
    
    C = []
    tuplesC = []
    csize = 0
    
    for i in range(a):
        for j in range(b):
            term = multiplyPauliString(A[i],tuplesA[i],B[j],tuplesB[j])
            flag = 0
            for k in range(csize):
                if tuplesC[k]==term[1]:
                    flag = 1
                    C[k] = C[k]+term[0]
            if (flag == 0) & (abs(term[0])>accur):
                C.append(term[0])
                tuplesC.append(term[1])
                csize = csize + 1
    
    return C, tuplesC


def commutateLinCombWithoutFactorOf2(A,tuplesA,B,tuplesB,accur):
    a = len(A)
    b = len(B)
    
    C = []
    tuplesC = []
    csize = 0
    
    for i in range(a):
        for j in range(b):
            term = commutatePauliString(A[i],tuplesA[i],B[j],tuplesB[j])
            flag = 0
            for k in range(csize):
                if tuplesC[k]==term[1]:
                    flag = 1
                    C[k] = C[k]+term[0]/2
            if (flag == 0) & (abs(term[0])>accur):
                C.append(term[0]/2)
                tuplesC.append(term[1])
                csize = csize + 1
    
    return C, tuplesC


def cleancoefs(coefs, accur):
    '''Rounds coefficients that are smaller than accur to zero.
    '''
    
    for i in range(len(coefs)):
        if abs(coefs[i])<accur:
            coefs[i] = 0