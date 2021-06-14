# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:45:19 2020
Methods to minimize in order to find the coefficients in the Cartan Decomposition

Two Sections:
    AdjointRepNorm: Find the zero of |Ad_k(h) - H |, and optimizes over coefficients for k and h
    AdjointInnerProduct: Find a local minimum of <v, Ad_K(H)>, where v is a dense vector in h
@author: Thomas Steckmann
@author: Efekan Kokcu
"""
import numpy as np
import numpy.linalg as la
import math

from src.PauliOps import commutatePauliString, multiplyLinComb, simplifyLinComb



""" ADJOINT REPRESENTATION METHODS """

"""
Find the zero of |Ad_k(h) - H |, and optimizes over coefficients for k and h

Args:
    initialGuess: NumPyArray 
        NumPyArray organized as the coeffcients in order: [k + h] or [k_1 + h' + k_2 + h] or [k_1 + h'' + k_2  +  h'  +  k_3 + h'' + k_4   +   h]
        length (lenh + lenK) of the guess coefficients
    *args: Tuple
        depth: int
            number of iterations for the Adjoint Action Power series
        H: NumPyArray
            Hamiltonian coefficients in the hm basis
        hm: List of Tuples
            List of m Pauli String elements, with the h elements first
        k: List of List of Tuples
            List of k Pauli String elements, organized as [k, h'', h'] or [k, h''', h'',h'] algebra Tuple Lists
        h: List of Tuples
            List of h Pauli String elements
        
Return:
    norm: Float
        |Ad_k(h) - H |
"""
def adjointRepNorm(initialGuess, *args):
    

    #strips the parameters from *args so we can use in in the sciPy optimizer
    depth, H, hm, k, h = args

    #Number of parameters to optimize over
    lenh = len(h)
    lenk = [len(a) for a in k] #Example: for XY 4 site open, gives [4]. For XY 6 site open, gives [4, 4] corresponding to [len(k), len(h')]
    lenhm = len(hm)
    
    
    #The input might be multiple sets within a larger k. For instance, k might be decomposed as k1'h'k2'= k.
    #For this, we need two sets, h' and k', and three sets of coefficients, k1', h', and k2'
    
    #Builds the list of k algebra Tuples, to get [k,h,k, h, k,h,k, h...]
    #Steps as: [] + k + [] => [k] + [h''] + [k] => [kh''k] + [h'] + [kh''k] => [kh''kh'kh''h] 
    kAlgebraList = []
    print('DEBUYGGING')
    print(kAlgebraList)
    print(k)
    print()
    for stepIndex in range(len(k)):
        kAlgebraList.append(kAlgebraList + k[stepIndex] + kAlgebraList)

    print(kAlgebraList)
    
    #Splits the h and k guess from the guess, k is complex but we passed an int so we fix that

    allkCoefficients = initialGuess[lenh:] * 1j
    hinitialGuess = initialGuess[:lenh]
    
    #Further splits k into sections:
    kCoefficientsList = []
    buildStepIndices = [] #Temporary array to build up the indices needed to pull apart the input k coefficients
    for stepIndex in range(len(k)):
        buildStepIndices = (buildStepIndices + [lenk[stepIndex]] + buildStepIndices)
    buildStepIndices = [0] + buildStepIndices
    print(buildStepIndices)
    buildIndices = [sum(buildStepIndices[:i]) for i in range(len(buildStepIndices))]
    print(buildIndices)
    for i in range(buildIndices - 1):
        kCoefficientsList.append(allkCoefficients[buildIndices[i]:buildIndices[i+1]])
    
        
    




    """    
    #strips the parameters from *args so we can use in in the sciPy optimizer
    depth, H, hm, k, h = args

    #Number of parameters to optimize over
    lenh = len(h)
    lenk = len(k)
    lenhm = len(hm)
    
    
    #The input might be multiple sets within a larger k. For instance, k might be decomposed as k1'h'k2'= k.
    #For this, we need two sets, h' and k', and three sets of coefficients, k1', h', and k2'


    #Strips the h and k guess from the guess, k is complex but we passed an int so we fix that
    hinitialGuess = initialGuess[:lenh]
    
    kinitialGuess = initialGuess[lenh:] * 1j
    """
    #Padds out empty m terms onto the h Guess so that we can evaluated it using properly sized matricies.  
    CalcualtedM = np.concatenate((hinitialGuess, np.zeros(lenhm - lenh,dtype=np.complex128)))


    for protoIndex in range(len(kAlgebraList)):
        kAlgebraIndex = len()
        #adMap generates the ad_k matrix. See the documentation in the adjointAction.py
        adMap = adjointAction(hm, kAlgebraList[kAlgebraIndex], kCoefficientsList[kAlgebraIndex]) #*2 #########WARNING: I think I fixed this? #########################################################################
    
        #The part where we evaluate the resultant Pauli string, CalcualtedM. Repeats depending on the desired depth of adjoint action operations to estimate the adjoint representation
        #Uses BCH lemma to calculated the value of Ad_k without actually doing any exponentialtion. Basically just a taylor series for the exponential
        CalcualtedM = adjointRep(depth, adMap, CalcualtedM, lenhm)
    

    #Distance from the Hamiltonian
    F = CalcualtedM - H
    norm = la.norm(F)
    
    return norm


"""
Computes the adjoint action operator. 
This is a matrix which contains the Structure constants of each element in m commuted with each element in k
Args:
    hmTupleList: List of Tuples
        The list of PauliString tuples of all the elements in the M algebra.
        Ordered so that the first elements are in h, followed by the remaining elements of m
    kTupleList: List of Tuples
        The list of PauliString tuples of the elements of the k algebra
    kVector: list of numbers
        The vector representation of the k element which generates ad_k.
        A list of coefficients with indices corresponding to the PauliStrings in kTupleList
Return:
    adjointMap: NumPyArray
        Numpy array of the ad_k map
Raises:
    Exception:
        Mismatched input lengths
"""
def adjointAction(hmTupleList,kTupleList,kVector):

    lenK = len(kTupleList)
    lenkVector = len(kVector)
    lenHM = len(hmTupleList)
    if lenK != lenkVector:
        raise Exception("Mismatched K Vector length")
    
    
    adjointMap = np.zeros((lenHM,lenHM),dtype=np.complex128) #populates the output with a zero array
    
    for mCounter in range(lenHM):
        for indexK in range(lenK):
            kTuple = kTupleList[indexK]
            mTuple = hmTupleList[mCounter]
            coefficient, targetTuple = commutatePauliString(1, kTuple, 1, mTuple)
            if coefficient == 0:
                pass
            else:
                #The Index in the tuple in the m vector. Actually an integer Index, that gives the location in the matrix
                hListIndex = hmTupleList.index(targetTuple)
                #hListIndex = 1
                adjointMap[(hListIndex, mCounter)] = coefficient * kVector[indexK]
    return adjointMap


"""
Calculates the Adjoint Representation using the BCH lemma using the adjoint action

Args: 
    depth: int
        number of iterations in the taylor series
    ad: ndarray
        the matrix represenation of the adjoint action acting on the m space
    hGuess: numpy array
        the vector represenation of h in the hm basis
    lenhm: int
        the number of elements in the m basis

Return:
    CalculatedM: numpy array
        the resultant vector in hm basis
"""
def adjointRep(depth, ad, hGuess, lenhm):
    
    operatorList = [np.diag(np.ones(lenhm,dtype=np.complex128))]
    for n in range(depth-1):
        if n == 0:
            operatorList.append(operatorList[n] @ ad)
        else:
            operatorList.append(operatorList[n] @ ad / (n+1))
    sumAOperators = sum(operatorList)
    
    #Calculates the resulting vector in m = KhK\dagger
    CalcualtedM = sumAOperators @ hGuess
    
    
 