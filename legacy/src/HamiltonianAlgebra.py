# -*- coding: utf-8 -*-
__docformat__ = "google"
"""
Created on Mon Dec 21 15:52:59 2020

Methods to find the Hamiltonian for a specific model.  Due to the nature of choosing the model, each model type is specifically defined

Usage: 
    To get the Hamiltonian Tuple elements
    To get the k,h,m,H elements for a given model type

@author: Thomas Steckmann
@author: Efekan Kokcu
"""

import numpy as np

from src.CartanMethods import makeGroup, elemcount, evenOdd, getsubalgebraelem



def generateAlgebra(sites, modelType, J, closed, returnAlg = False, manualMode = False):
    """Provided the model parameters, generates the Hamiltonian and the decomposition elements
    
    Currently Implemented Models:
     - XY: (XX + YY)
     - XX: (XXI + IXX)
     - YY: (YYI + IYY)
     - ZZ: (ZZI + IZZ)
     - Transverse_Z: (IZ + ZI)
     - KitaevEven: (XXI + IYY)
     - KitaevOdd: (YYI + IXX)
     - TransverseIsing: (XX + IZ + ZI)
     - Heisenberg: (XX + YY + ZZ)
    
     Generates the Hamiltonian, a Cartan Decomposition, and the Pauli String algebraic elements



    Args:
        sites (int): 
            Number of qubits to use (lattice points)
        modelType (Tuple of Strings): 
            Format: ('modelNameOne','modelNameTwo', ...). 
            
            For a single model, use: ('modelNameOne',)
                Currently Implemented Models:
                - 'XY'
                - 'XX' 
                - 'YY'
                - 'ZZ' 
                - 'Transverse_Z' 
                - 'KitaevEven' 
                - 'KitaevOdd' 
                - 'TransverseIsing' 
                - 'Heisenberg'
                - [(List of Pauli String Tuples)]: Must include (),manualMode=True) to use. Pauli String List formatted as H = XZYI + XXZZ written as [(1,3,2,0),(1,1,3,3)]
        J (Tuple of floats): 
            The Coupling constants.

            Two Possible Formats:
                - `(1,1)`: Two models, each with coefficient 1 on each term
                - `([1,0.9,0.8,...],[...])`: Two models, with coefficients ordered for each term
        closed (Bool): True if the model has periodic boundary condistions
            returnAlg (bool, optional): 
            True if the return should be formatted as a list of Tuples formatted as: `[(1, (1,1,0,0)), ...]` would be `H = 1*XXII`

                Default: Return the Coefficients as a vector indexed by the terms in the h algebra + the m / h algebra
                
                returnAlg = True should be used with the EarpPachosInnerProduct method
        manualMode (bool, optional): Allows for explicit input of a not implemented Hamiltonian

    """
    if not manualMode:
        if not isinstance(modelType, tuple):
            raise Exception("Model Type must be a Tuple")
        if not isinstance(J, tuple):
            raise Exception("Coupling (J) must be a Tuple")
        H = []
    
        newJ = [[] for a in J]
        for i in range(len(modelType)):
            toAdd = hamiltonianTuples(sites, modelType[i], closed)
            H = H + toAdd
    
            if isinstance(J[i], list):
    
                if len(J[i]) != len(toAdd):
                    raise Exception("Coupling Mismatch")
            else:
                newJ[i] = [J[i] for a in toAdd]
        J = newJ
    else:
        H = modelType
        newJ = [[] for a in J]
        for i in range(len(modelType)):
            if isinstance(J[i], list):
                if len(J[i]) != len(modelType):
                    raise Exception("Coupling Mismatch")
            else:
                newJ[i] = [J[i] for a in toAdd]

    g = makeGroup(H)

    if "TransverseIsing" in modelType or "Transverse_Z" in modelType:
        k,m = elemcount(g,2)
    else:
        k,m = evenOdd(g)


    h = getsubalgebraelem(m,[H[0]])

    hm = []
    for item in h:
        hm.append(item)
    for item in m:
        if not item in hm:
            hm.append(item)
    #Verifies the validity of the h solution
    if (len(h) + len(k) != len(m)):
        print('%s (h) + %s (k) != %s (m)' % (len(h), len(k), len(m)))
        raise Exception("Invalid h Guess. len(h) + len(k) != len(m)")
    #Populates the Tuple list of hm with the Hamiltonian coefficients
    hamiltonian = np.zeros(len(m))
    hamAlgList = []
    
    for i in range(len(modelType)):
        Hi = hamiltonianTuples(sites, modelType[i], closed)
        
        for j in range(len(Hi)):
            hamiltonian[hm.index(Hi[j])] = J[i][j]
            hamAlgList.append((J[i][j], Hi[j]))
    
    #plt.legend()
    #plt.suptitle(modelType[0])
    #plt.show()
    if returnAlg:    
        return (hamAlgList, hm, k, h)
    else: 
        return (hamiltonian, hm, k, h)


def hamiltonianTuples(sites, modelType, closed):
    """Helper Function for generateAlgebra.
    
    From the sites and model type, generates the Hamiltonian Pauli Strings

    Args: 
        sites (int): 
            Number of qubits in the system or lattice points.
        modelType (Tuple of Strings): 
            See generateAlgebra for formatting.
        closed (bool): 
            True if the model is period.

    """
    hamiltonianAlgebra = []
    if modelType == "XX": #XXII + IXXI + IIXX
        for i in range(sites - 1):
            term = ((0,) * i) + ((1,) * 2) + ((0,) * (sites - 2 - i))
            hamiltonianAlgebra.append(term)
        if  closed:
            term = (1,) + ((0,) * (sites - 2)) + (1,)
            hamiltonianAlgebra.append(term)

        
        
    elif modelType == "YY": #YYII + IYYI + IIYY
        for i in range(sites - 1):
            term = ((0,) * i) + ((2,) * 2) + ((0,) * (sites - 2 - i))
            hamiltonianAlgebra.append(term)
        if  closed:
            term = (2,) + ((0,) * (sites - 2)) + (2,)
            hamiltonianAlgebra.append(term)

        
        
    elif modelType == "ZZ": #ZZII + IZZI + IIZZ
        for i in range(sites - 1):
            term = ((0,) * i) + ((3,) * 2) + ((0,) * (sites - 2 - i))
            hamiltonianAlgebra.append(term)
        if  closed:
            term = (3,) + ((0,) * (sites - 2)) + (3,)
            hamiltonianAlgebra.append(term)
            
    elif modelType == "XY": #XXII + YYII + IXXI + IYYI + IIXX + IIYY
        for i in range(sites - 1):
            termX = ((0,) * i) + ((1,) * 2) + ((0,) * (sites - 2 - i))
            termY = ((0,) * i) + ((2,) * 2) + ((0,) * (sites - 2 - i))
            hamiltonianAlgebra.append(termX)
            hamiltonianAlgebra.append(termY)
        if  closed:
            termX = (1,) + ((0,) * (sites - 2)) + (1,)
            termY = (2,) + ((0,) * (sites - 2)) + (2,)
            hamiltonianAlgebra.append(termX)
            hamiltonianAlgebra.append(termY)
        
            """
    Blocked out because it is easier to do two Kitaev models and then merge them (half the parameters and a nicer solution)
    elif modelType == "XY": #XXII + YYII + IXXI + IYYI + IIXX + IIYY
        for i in range(sites - 1):
            termX = ((0,) * i) + ((1,) * 2) + ((0,) * (sites - 2 - i))
            termY = ((0,) * i) + ((2,) * 2) + ((0,) * (sites - 2 - i))
            hamiltonianAlgebra.append(termX)
            hamiltonianAlgebra.append(termY)
        if  closed:
            termX = (1,) + ((0,) * (sites - 2)) + (1,)
            termY = (2,) + ((0,) * (sites - 2)) + (2,)
            hamiltonianAlgebra.append(termX)
            hamiltonianAlgebra.append(termY) 
            """
        
    elif modelType == "XY2D": #XXII + YYII + IXXI + IYYI + IIXX + IIYY
        print("WARNING: THIS IS BROKEN")
        for i in range(sites - 1):
            for j in range(sites-1):
                termX1 = (0,) * sites * j +  (0,) * i \
                    + ((1,) * 2) + ((0,) * (sites - 2 - i)) \
                    + (0,) * sites * (sites - 1 - j) 
                        
                termX2 = (0,) * sites * j \
                    + (0,) * i + (1,)  + (0,) * (sites - i - 1) \
                    + (0,) * i + (1,)  + (0,) * (sites - i - 1) \
                    + (0,) * sites * (sites - j - 2)
                         
                termY1 = (0,) * sites * j \
                    + (0,) * i + ((2,) * 2) + ((0,) * (sites - 2 - i)) \
                    + (0,) * sites * (sites - 1 - j) 
                        
                termY2 = (0,) * sites * j \
                    + (0,) * i + (2,)  + (0,) * (sites - i - 1) \
                    + (0,) * i + (2,)  + (0,) * (sites - i - 1) \
                    + (0,) * sites * (sites - j - 2)

                hamiltonianAlgebra.append(termX1)
                hamiltonianAlgebra.append(termX2)
                hamiltonianAlgebra.append(termY1)
                hamiltonianAlgebra.append(termY2)

                
        if  closed:
            for j in range(sites - 1):
                termX1 = (0,) * sites * j \
                    + (1,) +  (0,) * (sites - 2) + (1,) \
                    + (0,) * sites * (sites - 1 - j) 
                        

                         
                termY1 = (0,) * sites * j \
                    + (2,) +  (0,) * (sites - 2) + (2,) \
                    + (0,) * sites * (sites - 1 - j) 
                        
 
                hamiltonianAlgebra.append(termX1)
                hamiltonianAlgebra.append(termY1)

            for i in range(sites - 1):
                termX2 = (0,) * i + (1,)  + (0,) * (sites - i - 1) \
                    + (0,) * 3 \
                    + (0,) * i + (1,)  + (0,) * (sites - i - 1) 
                         
                termY2 = (0,) * i + (2,)  + (0,) * (sites - i - 1) \
                    + (0,) * 3 \
                    + (0,) * i + (2,)  + (0,) * (sites - i - 1) 
                         
                hamiltonianAlgebra.append(termX2)
                hamiltonianAlgebra.append(termY2)

        
    elif modelType == "KitaevEven": #XXII + IYYI + IIXX
        for i in range(sites - 1):
            if i % 2 == 0:
                term = ((0,) * i) + ((1,) * 2) + ((0,) * (sites - 2 - i))
            else:
                term = ((0,) * i) + ((2,) * 2) + ((0,) * (sites - 2 - i))
            hamiltonianAlgebra.append(term)
        if  closed:
            term = (2,) + ((0,) * (sites - 2)) + (2,)
            hamiltonianAlgebra.append(term)

        
        
    elif modelType == "KitaevOdd": #YYII + IXXI + IIYY
        for i in range(sites - 1):
            if i % 2 == 0:
                term = ((0,) * i) + ((2,) * 2) + ((0,) * (sites - 2 - i))
            else:
                term = ((0,) * i) + ((1,) * 2) + ((0,) * (sites - 2 - i))
            hamiltonianAlgebra.append(term)
        if  closed:
            term = (1,) + ((0,) * (sites - 2)) + (1,)
            hamiltonianAlgebra.append(term)



    elif modelType == "Heisenberg": #XXII + YYII + ZZII + ...
        for i in range(sites - 1):
            termX = ((0,) * i) + ((1,) * 2) + ((0,) * (sites - 2 - i))
            termY = ((0,) * i) + ((2,) * 2) + ((0,) * (sites - 2 - i))
            termZ = ((0,) * i) + ((3,) * 2) + ((0,) * (sites - 2 - i))
            hamiltonianAlgebra.append(termX)
            hamiltonianAlgebra.append(termY)
            hamiltonianAlgebra.append(termZ)
        if  closed:
            termX = (1,) + ((0,) * (sites - 2)) + (1,)
            termY = (2,) + ((0,) * (sites - 2)) + (2,)
            termY = (3,) + ((0,) * (sites - 2)) + (3,)
            hamiltonianAlgebra.append(termX)
            hamiltonianAlgebra.append(termY)
            hamiltonianAlgebra.append(termZ)

        

    elif modelType == "TransverseIsing": # XXII + IXXI + IIXX + ZIII + IZII + IIZI + IIIZ
        #raise Exception("Warning: TransverseIsing Involution not yet implemented, so m construction will fail")
        for i in range(sites - 1):
            term = ((0,) * i) + ((1,) * 2) + ((0,) * (sites - 2 - i))
            hamiltonianAlgebra.append(term)
        if  closed:
            term = (1,) + ((0,) * (sites - 2)) + (1,)
            hamiltonianAlgebra.append(term)
        for i in range(sites):
            term = ((0,) * i) + (3,) + ((0,) * (sites - 1 - i))
            hamiltonianAlgebra.append(term)



    elif modelType == "Transverse_Z": # XXII + IXXI + IIXX + ZIII + IZII + IIZI + IIIZ
        for i in range(sites):
            term = ((0,) * i) + (1,) + ((0,) * (sites - 1 - i))
            hamiltonianAlgebra.append(term)
        
        
        
    else:
        raise Exception("Invalid Model. Valid models include: 'XX', 'YY', 'ZZ', 'XY', 'KitaevEven', 'KitaevOdd', 'Heisenberg', TransverseIsing', and 'Transverse_Z'")
    #print(hamiltonianAlgebra)
    return hamiltonianAlgebra
                    