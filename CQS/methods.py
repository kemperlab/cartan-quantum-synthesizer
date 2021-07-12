# -*- coding: utf-8 -*-
__docformat__ = 'google'

#General Imports
import time
import scipy.optimize
import numpy as np
import csv
import os
import math

#Local Imports
from CQS.util.PauliOps import commutatePauliString
import CQS.util.IO as IO
from CQS.util.IO import printlist, paulilabel
import CQS.util.verification


RULES = [1,3,1,3]
#Used in generating commutator tables. See util.PauliOps
SIGN_RULES = [[1,1,1,1], 
             [1, 1, 1j, -1j],
             [1, -1j, 1, 1j],
             [1, 1j, -1j, 1]]
#Used in generating commutator tables. See util.PauliOps

class Hamiltonian:
    """
    Class contains information about the Hamiltonian to be decomposed.

    ## Functionality:
    * Build Hamiltonians from prebuilt Spin models
    * Add custom Hamiltonians 

    TODO:
    * Add Hubbard Model

    Authors:
    * Thomas Steckmann
    * Efekan Kokcu
    """
    def __init__(self, sites, name = None):
        """Initializes an emtpy Hamiltnoan, unless name is specified

        name is not required as an input to initialize the Hamiltonian, only the number of qubits. After intializing the Hamiltonian, users can use addModel() to add a single, imlemented Hamiltonian at a time and use .addTerms() to add lists of terms or individual terms. .removeTerm() can be used to remove a single term at a time

        Args:
            qubits (int): The number of lattice points the Hamiltonian exists on
            name (List of Tuples): Easily build Hamiltonians from native models. 

                ### Options:
                    - `[(coefficient, 'modelname', periodic boundary condisitons boolian)]` - Builds a single Hamiltonian with constant coefficient. Periodic boundary conditions are true or false, false is assumed if not specified
                    - `[([list of coefficients], 'modelname')]` - Populates the Hamiltonian using the list of coefficients. Lengths must match
                    - `[(coefficient, 'modelname1'),([list of coefficients], 'modelname2')]` - Combines two Hamiltonains. Order my alter default choice of 𝖍
                #### Examples:
                    - `[(1,'xy', True)]` on three qubits gives: XXI + YYI + IXX + IYY + XIX + YIY
                    - `[([1,2],'kitaevEven')]` on three qubits gives: 1*XXI + 2*IYY 
                ### Currently Implemented Models:
                    - xy: (XX + YY)
                    - xx: (XXI + IXX)
                    - yy: (YYI + IYY)
                    - zz: (ZZI + IZZ)
                    - tfim: (ZZ + XI + IX)
                    - tfxy: (XX + YY + ZI + IZ)
                    - transverse_z: (IZ + ZI)
                    - kitaev_even: (XXI + IYY)
                    - kitaev_odd: (YYI + IXX)
                    - heisenberg: (XX + YY + ZZ)
                    - TODO: hubbard: (XXII + YYII + IIXX + IIYY + ZIZI + IZIZ)
        """
        self.sites = sites
        self.HTuples = []
        self.HCoefs = []
        #Builds the Hamiltonian
        if name is not None:
            for pair in name: 
                self.addModel(pair) 

    def addModel(self, pair):
        """Adds a predefined model to the Hamiltonian object
        
        Args:
            pair (Tuple):
                
                ### Currently Implemented Models:
                    - xy: (XX + YY)
                    - xx: (XXI + IXX)
                    - yy: (YYI + IYY)
                    - zz: (ZZI + IZZ)
                    - tfim: (ZZ + XI + IX)
                    - tfxy: (XX + YY + ZI + IZ)
                    - transverse_z: (IZ + ZI)
                    - kitaev_even: (XXI + IYY)
                    - kitaev_odd: (YYI + IXX)
                    - heisenberg: (XX + YY + ZZ)
                    - TODO: hubbard: (XXII + YYII + IIXX + IIYY + ZIZI + IZIZ)
        
        Raises:
            ValueError:  Coefficient list length mismatch
            Exception: Invalid model type input
        """
        if len(pair) == 3: #Handles the boundary condition variable format
            hamiltonianTerms = self.generateHamiltonian(pair[1], closed=pair[2]) #pair[1] = modelname, pair[2] = boundary condition
            if isinstance(pair[0], list): #pair[0] = coefficients float or list
                if len(pair[0]) != len(hamiltonianTerms):
                     raise ValueError("Coefficient List mismatch: Expected {} coefficients".format(len(hamiltonianTerms)))
                pairedIterable = zip(pair[0],hamiltonianTerms)
                for (coefficient, pauliString) in pairedIterable:
                    self.HTuples.append(pauliString)
                    self.HCoefs.append(coefficient)
            elif pair[0]:
                coefficient = pair[0]
                for pauliString in hamiltonianTerms:
                    self.HTuples.append(pauliString)
                    self.HCoefs.append(coefficient)
        else:
            hamiltonianTerms = self.generateHamiltonian(pair[1]) #pair[1] = modelname,
            if isinstance(pair[0], list): #pair[0] = coefficients float or list
                if len(pair[0]) != len(hamiltonianTerms):
                     raise ValueError("Coefficient List mismatch: Expected {} coefficients".format(len(hamiltonianTerms)))
                pairedIterable = zip(pair[0],hamiltonianTerms)
                for (coefficient, pauliString) in pairedIterable:
                    self.HTuples.append(pauliString)
                    self.HCoefs.append(coefficient)
            elif pair[0]:
                coefficient = pair[0]
                for pauliString in hamiltonianTerms:
                    self.HTuples.append(pauliString)
                    self.HCoefs.append(coefficient)

    def addTerms(self, pairs):
        """Adds custom elements to build out the Hamiltonian

        Args:
            pair (List of Tuples or Tuple of Lists or Tuple): Formatted as either
                * `[(coefficient, PauliString), (coefficient, PauliStringTuple),...]`
                * `(coefficient, (PauliString))`
                * `([coefficienList],[(PauliStringList)])

        Examples:
            * `Hamiltonian.addTerms((0.45, (1,3,2,0,0,0)))`
            * `Hamiltonian.addTerms([(1,(1,1,0,0)),(2,(2,2,0,0))])
            * `Hamiltonian.addTerms(([co1, co2, co3, ...],[(PauliString1),(PauliString2),(PauliString3)...])`

        """
        if isinstance(pairs, list): #Verifies the User input - if a list, iterates over it
            for pair in pairs:
                try: #Checks if the element is already in the list and adds it
                    index = self.HTuples.index(pair[1])
                    self.HCoefs[index] = self.HCoefs[index] + pair[0]
                except: #If not in the list, catches warning and adds the new term
                    self.HTuples.append(pair[1])
                    self.HCoefs.append(pair[0])
        elif isinstance(pairs, tuple): #Verifies user input - if a tuple, just adds it
            if isinstance(pairs[0], list):
                for (co, tup) in zip (pairs[0], pairs[1]):
                    try: #Checks if the element is already in the list and adds it
                        index = self.HTuples.index(tup)
                        self.HCoefs[index] = self.HCoefs[index] + co
                    except: #If not in the list, catches warning and adds the new term
                        self.HTuples.append(tup)
                        self.HCoefs.append(co)
            else:
                try: #Checks if the element is already in the list and adds it
                    index = self.HTuples.index(pairs[1])
                    self.HCoefs[index] = self.HCoefs[index] + pairs[0]
                except: #If not in the list, catches warning and adds the new term
                    self.HTuples.append(pairs[1])
                    self.HCoefs.append(pairs[0])

    def removeTerm(self, tup):
        """Removes the term matching the tuple input. Used for trimming terms off the Hamiltonian
        Args:
            tup (tuple): A (pauliString)
        """
        index = self.HTuples.index(tup)
        
        self.HTuples.pop(index)
        self.HCoefs.pop(index)

    def generateHamiltonian(self, modelType, closed = False):
        """Helper Function to generate Hamiltonians.
        
        From the model type, generates the Hamiltonian Pauli Strings

        Args: 
            modelType (Tuple of Strings): See previous documentation
            closed (bool): 
                True if the model is period.

        """
        hamiltonian = []

        if modelType == "xx": #XXII + IXXI + IIXX
            for i in range(self.sites - 1):
                term = ((0,) * i) + ((1,) * 2) + ((0,) * (self.sites - 2 - i))
                hamiltonian.append(term)
            if  closed:
                term = (1,) + ((0,) * (self.sites - 2)) + (1,)
                hamiltonian.append(term)

            
            
        elif modelType == "yy": #YYII + IYYI + IIYY
            for i in range(self.sites - 1):
                term = ((0,) * i) + ((2,) * 2) + ((0,) * (self.sites - 2 - i))
                hamiltonian.append(term)
            if  closed:
                term = (2,) + ((0,) * (self.sites - 2)) + (2,)
                hamiltonian.append(term)

            
            
        elif modelType == "zz": #ZZII + IZZI + IIZZ
            for i in range(self.sites - 1):
                term = ((0,) * i) + ((3,) * 2) + ((0,) * (self.sites - 2 - i))
                hamiltonian.append(term)
            if  closed:
                term = (3,) + ((0,) * (self.sites - 2)) + (3,)
                hamiltonian.append(term)


        elif modelType == "xy": #XXII + YYII + IXXI + IYYI + IIXX + IIYY
            for i in range(self.sites - 1):
                termX = ((0,) * i) + ((1,) * 2) + ((0,) * (self.sites - 2 - i))
                termY = ((0,) * i) + ((2,) * 2) + ((0,) * (self.sites - 2 - i))
                hamiltonian.append(termX)
                hamiltonian.append(termY)
            if  closed:
                termX = (1,) + ((0,) * (self.sites - 2)) + (1,)
                termY = (2,) + ((0,) * (self.sites - 2)) + (2,)
                hamiltonian.append(termX)
                hamiltonian.append(termY)

            
        elif modelType == "kitaev_even": #XXII + IYYI + IIXX
            for i in range(self.sites - 1):
                if i % 2 == 0:
                    term = ((0,) * i) + ((1,) * 2) + ((0,) * (self.sites - 2 - i))
                else:
                    term = ((0,) * i) + ((2,) * 2) + ((0,) * (self.sites - 2 - i))
                hamiltonian.append(term)
            if  closed:
                term = (2,) + ((0,) * (self.sites - 2)) + (2,)
                hamiltonian.append(term)

            
            
        elif modelType == "kitaev_odd": #YYII + IXXI + IIYY
            for i in range(self.sites - 1):
                if i % 2 == 0:
                    term = ((0,) * i) + ((2,) * 2) + ((0,) * (self.sites - 2 - i))
                else:
                    term = ((0,) * i) + ((1,) * 2) + ((0,) * (self.sites - 2 - i))
                hamiltonian.append(term)
            if  closed:
                term = (1,) + ((0,) * (self.sites - 2)) + (1,)
                hamiltonian.append(term)



        elif modelType == "heisenberg": #XXII + YYII + ZZII + ...
            for i in range(self.sites - 1):
                termX = ((0,) * i) + ((1,) * 2) + ((0,) * (self.sites - 2 - i))
                termY = ((0,) * i) + ((2,) * 2) + ((0,) * (self.sites - 2 - i))
                termZ = ((0,) * i) + ((3,) * 2) + ((0,) * (self.sites - 2 - i))
                hamiltonian.append(termX)
                hamiltonian.append(termY)
                hamiltonian.append(termZ)
            if  closed:
                termX = (1,) + ((0,) * (self.sites - 2)) + (1,)
                termY = (2,) + ((0,) * (self.sites - 2)) + (2,)
                termY = (3,) + ((0,) * (self.sites - 2)) + (3,)
                hamiltonian.append(termX)
                hamiltonian.append(termY)
                hamiltonian.append(termZ)

            

        elif modelType == "tfim": # ZZII + IZZI + IIXZZ + XIII + IXII + IIXI + IIIX
            #raise Exception("Warning: TransverseIsing Involution not yet implemented, so m construction will fail")
            for i in range(self.sites - 1):
                term = ((0,) * i) + (3,3) + ((0,) * (self.sites - 2 - i))
                hamiltonian.append(term)
            if  closed:
                term = (3,) + ((0,) * (self.sites - 2)) + (3,)
                hamiltonian.append(term)
            for i in range(self.sites):
                term = ((0,) * i) + (1,) + ((0,) * (self.sites - 1 - i))
                hamiltonian.append(term)

        elif modelType == 'tfxy':
            for i in range(self.sites):
                term = (0,)*i + (3,) + (0,)*(self.sites-i-1) 
                hamiltonian.append(term)
            
            for i in range(self.sites-1):
                term = (0,)*i + (1,1) + (0,)*(self.sites-i-2) 
                hamiltonian.append(term)
                term = (0,)*i + (2,2) + (0,)*(self.sites-i-2) 
                hamiltonian.append(term)
                
            if closed:
                term = (1,) + (0,)*(self.sites-2) + (1,)
                hamiltonian.append(term)
                term = (2,) + (0,)*(self.sites-2) + (2,)
                hamiltonian.append(term)
            

        elif modelType == "transverse_z": # ZIII + IZII + IIZI + IIIZ
            for i in range(self.sites):
                term = ((0,) * i) + (3,) + ((0,) * (self.sites - 1 - i))
                hamiltonian.append(term)

        else:
            raise Exception("Invalid Model. Valid models include: 'XX', 'YY', 'ZZ', 'XY', 'KitaevEven', 'KitaevOdd', 'Heisenberg', TransverseIsing', and 'Transverse_Z'")
        return hamiltonian
        
    def getHamiltonian(self, type='tuples'):
        """ Based on the type, returns the Hamiltonian from the object and formats it.
        Args:
            type (String, default = 'tuples'): Specifies the return type of the function.
                Valid inputs:

                 - 'tuples': Return formatted as `[(coefficient, (PauliString), ... ]`

                 - 'printTuples': Prints out to the console ``(coefficient, (PauliString)) \n ...``

                 - 'printText': Prints out ``'coefficient * 'PauliString' + ....``

                 - 'text': Return formatted as a list of `[[coefficient, 'PauliString']]`
        
        Returns:
            For type=`'tuples'` or type=`'text'`, returns

            - `'tuples'`: A list of Coefficient, (PauliString) Tuple pairs
            - `'text'`: A List of Coefficient 'PauliString' List pairs
        
        Raises:
            ValueError: Invalid type
        """
        if type == 'tuples':
            returnVal = []
            for (co, tup) in zip(self.HCoefs, self.HTuples):
                returnVal.append((co,tup))
            return returnVal
        elif type == 'printTuples':
            for (co, tup) in zip(self.HCoefs, self.HTuples):
                print((co,tup))
        elif type == 'printText':
            for (co, tup) in zip(self.HCoefs, self.HTuples):
                print(str(co) + " * " + str(paulilabel(tup))) #Prints coefficient'pauliString'
        elif type == 'text':
            returnlist = []
            for (co, tup) in zip(self.HCoefs, self.HTuples):
                returnlist.append([co,  str(paulilabel(tup))]) #Prints coefficient'pauliString'
            return returnlist
        else:
            raise ValueError('Invalid Type: Must be "tuples", "printTuples", "printText", or "text"')

class Cartan:
    """
    Class to contain the options for performing Cartan Decomposition on a Hamiltonian
        
    # Object containing the functions and data required to generate a Cartan Decomposition from a given Hamiltonian. 
    from object_based.PauliOps import commutatePauliString

    TODO:
    * Add in k simplification functions

    Authors:
    * Thomas Steckmann
    * Efekan Kokcu

    Functionality:
    * Generate Hamiltonian Algebra
    * Generate k, m, h partition
    * Choose involution (Default - Even/Odd)
    * Seed choice of h
    * Modify k (Additional Decomposition, TODO: Abelian Decomposition, Piling)
    """
    def __init__(self, hamObj, involution='evenOdd', order = 0, manualMode=0):
        """ Generates the Cartan Object
        
        Args:
            hamObj (Hamiltonian): Passes a Hamiltonian Object containing the full information about the system
            involution (String, default='evenOdd'):
                Allows a choice of the k,m involuiton

                Options:
                * `'evenOdd'`: m contains an even number of non-identity pauli terms in each string, k contains an odd nunber of non-idenity elements
                * `'knejaGlaser'`: m contains elements ending in Y or X, k contains elements ending in I or Z
                * `'count' + 'X', 'Y', or 'Z'`: Counts of the number of the specified Pauli Tuple. Even count in m, odd in k
            order (int, default = 0):
                g(H) is generated iteratively until it fails to generate new terms. We call each term in H, [H,H], [H,[H,H]], ... order 0, 1, 2, and going up.
                Generally, the larger the system the larger the order of terms required to generate g(H). Setting the order greater than 0 ensure that g(H) will only result in terms the distance of commutators from H.
                However, this results in an incomplete algebra for g. Be warned. 
            manualMode (bool, default = 0):
                Choose either 0 (automatic) or 1 (manual). automatic generates h, k, m, and h when the object is created. manual requires the user call the respective functions.
        Attributes:
            hamiltonian (::hamiltonian:: object): Allows access to HCoefficients and HTuples
            HTuples (List of Tuples): Copies over the HTuples from the hamiltonian object
            g (List of Tuples): Generates formatted like k + h + (m without h)
            k (List of Tuples): Specified by the decomposition. Changing k regenerates h and the order of g (g = k + h + (m\h))
            h (List of Tuples): Specified by SubAlgebra(). Defaults to seeding by m, otherwise allows for inclusion of specific elements
        """
        self.mode = manualMode
        self.hamiltonian = hamObj
        self.HTuples = self.hamiltonian.HTuples
        if self.mode == 0:
            self.g = self.makeGroup(self.HTuples, order)
            self.decompose(involution)
        

    def decompose(self, involutionName):
        """
        Sets a new Involution using a switch. Regenerates h using the default first element in m

        Options:
                * `'evenOdd'`: m contains an even number of non-identity pauli terms in each string, k contains an odd nunber of non-idenity elements
                * `'knejaGlaser'`: m contains elements ending in Y or X, k contains elements ending in I or Z
                * `'count' + 'X', 'Y', or 'Z'`: Counts of the number of the specified Pauli Tuple. Even count in m, odd in k
        """
        self.involution = involutionName #Store involution name

        if involutionName == 'evenOdd': 
            (self.m, self.k) = self.evenOdd(self.g)
        elif involutionName == 'knejaGlaser':
            (self.m, self.k) = self.knejaGlaser(self.g)
        elif involutionName == 'countX':
            (self.m, self.k) = self.elemcount(self.g, 1)
        elif involutionName == 'countY':
            (self.m, self.k) = self.elemcount(self.g, 2)
        elif involutionName == 'countZ': 
            (self.m, self.k) = self.elemcount(self.g, 3)
        invalidInvolutionFlag = 0
        for Hi in self.HTuples:
            if not Hi in self.m:
                invalidInvolutionFlag = 1
                break
        if invalidInvolutionFlag == 1:
            raise Exception('Invalid Involution. Please Choose an involution such that H ⊂ m')
        if self.mode == 0:
            self.subAlgebra()

    def makeGroup(self,g, order=0):
        '''
        Returns a closure of a given list of pauli strings (g). The list doesn't include any coefficients, it is just
        a tuple like (0,2,3) representing IYZ.

        Args:
            g (List of Tuples):
                A set of Pauli Strings
            order (int, default = 0): 
                The number of terms to include in g. If order = 0, generates the full g(H). 0th order is H, 1st is [H,H], and 2nd is [H,[H,H]], etc


        Returns:
            List of Tuples: 𝖌(H), the Hamiltonian Algebra generated by the input g
        '''
        if order == 0:
            flag = 0 
            while (flag == 0):
                flag = 1
                L = len(g) #g initializes as H
                #initialize commutations
                coms = [] #Tracks the commutations generated in this iteration

                #calculate all possible commutations 
                for i in range(L): #First index in the commutator
                    for j in range(i,L): #Second index is all terms after this in the list
                        m = commutatePauliString(1,g[i],1,g[j]) #generates the commutation
                        #add all new ones to the list, if they are not in the list and are not 0
                        if (abs(m[0])>0) & (self.included(coms,m[1])==0) & (self.included(g,m[1])==0):
                            #set flag to 0 whenever there is a new term to be added. If the flag remains 1, the loop terminates
                            flag = 0
                            coms.append(m[1])

                #then merge initial list with these new commutations
                g = g + coms
            return g
        else:
            loopOrder = 0
            flag = 0 
            while (flag == 0 and loopOrder <= order):
                flag = 1
                L = len(g) #g initializes as H
                #initialize commutations
                coms = [] #Tracks the commutations generated in this iteration

                #calculate all possible commutations 
                for i in range(L): #First index in the commutator
                    for j in range(i,L): #Second index is all terms after this in the list
                        m = commutatePauliString(1,g[i],1,g[j]) #generates the commutation
                        #add all new ones to the list, if they are not in the list and are not 0
                        if (abs(m[0])>0) & (self.included(coms,m[1])==0) & (self.included(g,m[1])==0):
                            #set flag to 0 whenever there is a new term to be added. If the flag remains 1, the loop terminates
                            flag = 0
                            coms.append(m[1])

                #then merge initial list with these new commutations
                loopOrder += 1
                g = g + coms
            return g

    
    def elemcount(self,g,element):
        '''
        Counts the number of given elements (X,Y or Z in number), and puts even numbers in m, odd numbers in k. 
        For element=2, it corresponds to θ(g) = -g^T
        '''    
        k = []
        m = []
        
        for i in range(len(g)):
            elem = g[i]
            count = 0
            for j in range(len(elem)):
                if elem[j] == element:
                    count=count+1
            if count%2 == 0:
                m.append(elem)
            else:
                k.append(elem)
            
        return m,k
    
    def included(self,g,m):
        '''Following function returns 0 if tuple m is not incu=luded in tuple list g, returns 1 if it is included.
            
            Args:
                g (List of Tuples): 
                    The List of Pauli string elements in the Hamiltonian Algebra 𝖌(H)
                m (Tuple of 0,1,2,3):
                    Pauli string in the set 𝖒

            Returns: 
                1 if m is in g
                0 if not
        '''
        L = len(g)
        res = 0    
        for i in range(L):
            if g[i]==m:
                res = 1
                break
        
        return res


    def evenOdd(self,g):
        """ Partitions the Algebra by counting the number of non-idenity Pauli elements

        Args:
            g (List of Tuples):
                The Algebra to partition
        
        Returns:
            k (List of Tuples):
                The List of Pauli Strings with Odd non-identity terms
            m (List of Tuples):
                The List of Pauli Strings with Even non-idenity terms
        """
        k = []
        m = []
        
        for i in range(len(g)):
            elem = g[i]
            count = 0
            for j in range(len(elem)):
                if elem[j] > 0:
                    count=count+1
            if count%2 == 0:
                m.append(elem)
            else:
                k.append(elem)
                
        return m,k

    def knejaGlaser(self,g):
        '''
        Following one is the Kneja Glaser decomposition. 
        It corresponds to \theta(g) = III...IZ g III..IZ
        ''' 
        k = []
        m = []
        
        for i in range(len(g)):
            elem = g[i]
            last = elem[len(elem)-1]
            if  (last == 3) | (last == 0):
                k.append(elem)
            else:
                m.append(elem)
            
        return m,k

    def subAlgebra(self,seedList=None):
        """ Generates h from a list of Commuting elements in the seedList 
        
        Args:
            seedList (List of Tuples): List of (PauliStrings). Must be commuting
        
        TODO:
            Verify that seedList is commuting
        """
        if seedList == None:
            seedList = [self.m[0]]
        h = seedList.copy()
        for i in range(len(self.m)):
            flag = 0
            for j in range(len(h)):
                if self.m[i] == h[j]:
                    flag = 1
                    break
                    
                comm = commutatePauliString(1, h[j], 1, self.m[i])
                if comm[0] != 0:
                    flag = 1
                    break
            if flag == 0:
                h.append(self.m[i]) 
                
        self.h = h
        #Regenerates the ordering for g, required to generate commutators

        m_tuples = self.m.copy()
        #Strips h from m
        index = 0
        while index < len(m_tuples): #
            flag = 0 #When flag = 1, 
            for i in range(len(self.h)):
                if m_tuples[index] == self.h[i]:
                    flag = 1
                    m_tuples.pop(index)
                    break
            if flag == 0:
                index = index + 1
        self.g = self.k + self.h + m_tuples #Reorder g

class FindParameters:
    """
    From a Cartan Decomposition, runs the optimizer to find the appropriate parameters for a circuit

    Warning: Very long runtime for more than 8ish qubits

    Authors:
    * Thomas Steckmann
    * Efekan Kokcu
    """
    def __init__(self, cartan, saveFileName = None, loadfileName=None, optimizerMethod='BFGS', accuracy=1e-5, initialGuess=None, steps = 5000, useCommTables=True):
        """
        Initializing a FindParameters class automatically runs the optimizer over the Cartan decomposition and provided Hamiltonian
        
        If saveFileName is provided, verifies file location before proceeding, and saves the output as a csv
        If loadFileName is provided, does not run the optimizer and instead reads data from the provided file

        TODO:
            * Write Load file functions
        
        Args:
            cartan (Obj Cartan): The Cartan object containing the k,h, and Hamiltonian information
            saveFileName (String, default=None): path to save the output. Do not include an extension ('csv' or 'txt')
            loadFileName (String, default=None): path to a csv containing previous optimizer results. Do not add .csv or .txt
            optimizer (String, default='BFGS'): The Scipy optimizer to use. Easy to add new ones, but must be done manually
                
                Options: 
                    * `'BFGS'` : Uses Gradient
                    * `'Powel'`: Does not use Gradient
            accuracy (float, default=1e-5): Optimizer convergence Criteria
            initialGuess (List of values): Allows the user to specify the inital guess for k. Must be correct, no input checking is currently implemented
            steps (int): The maximum number of optimization steps before termination
        
        Returns:
            * Progress updates and runtime predictions
            * prints the results of the optimization
        """
        #Initialize Values
        self.cartan = cartan #Extracts the hamiltonian object
        self.hamiltonian = self.cartan.hamiltonian #Extracts the hamiltonain objects
        self.useCommTables = useCommTables
        self.optimizerMethod = optimizerMethod
        self.accuracy = accuracy
        self.lenK = len(self.cartan.k)
        self.lenh = len(self.cartan.h)
        self.steps = steps
        #Begin Optimizer
        if loadfileName is not None: #If able to, loads prior results
            #with open(loadfileName + '.csv', "r") as f:
            #    csv_reader = csv.reader(csv_file, delimiter=',')
            raise Exception('Unable to continue, file loading is not implemented')
        elif self.lenK == 0: #Specific exception handling to prevent breaking on commuting Hamiltonians 
            self.hCoefs = self.hamiltonian.HCoefs
            #self.hTuples = self.hamiltonian.HTuples
            self.kCoefs = []
        else:   
            if saveFileName is not None:
                if not os.access(saveFileName, os.W_OK): #Verifies write permission
                    raise Exception('Save File location does not have write access. Aborting Optimization.')
                self.saveFileName = saveFileName
            
            #Generating reused values before begining the optimizer loop
            if useCommTables == True: #Faster, but only works for a complete algebra
                self.setCommutatorTables() #Sets values for a look-up table for the commutators
                self.generateIndexLists()
            #Generates coefficients for v, a dense element in the h algebra
            pi = np.pi
            self.vcoefs = [1]
            for i in range(len(self.cartan.h)-1):
                term = pi*self.vcoefs[i]
                while term > 1:
                    term = term-1
                self.vcoefs.append(term)
            

            
            #If specified, defines an inital guess. If not, sets it to all zeros
            if initialGuess is not None:
                self.initialGuess = initialGuess
            else:
                self.initialGuess = initialGuess = np.zeros(len(self.cartan.k))
            
            #Calls the function that contains the optimizer loop
            total_time = time.time()
            self.optimizerOutput = self.optimize()
            timepassed = time.time() - total_time
            print('--- ' + str(timepassed) +  ' seconds ---')

            self.sethVecFromk()
            self.error = self.errorhVec()

            print('Optimization Error:')
            print(self.error)

            if saveFileName is not None:
                self.saveKH()

    def setCommutatorTables(self):
        """ 
        Generates commutator look-up tables for the commutators of different elements

        Important: g must be ordered as k + h + (the rest of m). Does not work without this ordering
        """
        
        k_tuples = self.cartan.k.copy()
        h_tuples = self.cartan.h.copy()
        m_tuples = self.cartan.m.copy()
        
        index = 0
        
        while index < len(m_tuples):
            flag = 0
            for i in range(len(h_tuples)):
                if m_tuples[index] == h_tuples[i]:
                    flag = 1
                    m_tuples.pop(index)
                    break
            if flag == 0:
                index = index + 1
        
        g_tuples = k_tuples + h_tuples + m_tuples
        
        self.comm_table = np.zeros((len(g_tuples),len(g_tuples)))
        self.comm_coefs = np.zeros((len(g_tuples),len(g_tuples)),dtype=np.complex_)
        
        for i in range(len(g_tuples)):
            for j in range(len(g_tuples)):
                res = self.commutatePauliString(1,g_tuples[i],1,g_tuples[j])
                
                self.comm_coefs[i][j] = res[0]
                
                if res[0]==0:
                    self.comm_table[i][j]=0
                else:
                    for q in range(len(g_tuples)):
                        if res[1] == g_tuples[q]:
                            self.comm_table[i][j] = q
    
    def generateIndexLists(self):
        """Generates a lists for H, h, and k using indices in g instead of as lists of Tuples"""

        #Computes the indices of each of the k elements in the g list                   
        self.kElementIndices = []
        for i in range(len(self.cartan.k)):
            for j in range(len(self.cartan.g)):
                if self.cartan.k[i] == self.cartan.g[j]:
                    self.kElementIndices.append(j)
                    break  

        #computes the indicies of each of the Ham elements in the g list
        self.HElementIndices = [] 
        Htuples = self.hamiltonian.HTuples
        for i in range(len(Htuples)):
            for j in range(len(self.cartan.g)):
                if self.cartan.g[j]==Htuples[i]:
                    self.HElementIndices.append(j)
                    break
        
        #Computes the indices of the of the h elements in the g list:
        self.hElementIndices = [] 
        htuples = self.cartan.h
        for i in range(len(htuples)):
            for j in range(len(self.cartan.g)):
                if self.cartan.g[j]==htuples[i]:
                    self.hElementIndices.append(j)
                    break

    def optimize(self):
        """
        Chooses between methods of optimization. Current options are 'BFGS' and 'Powell' from scipy.optimize

        Sets the attribute kCoefs, which are the results of the optimizer

        Returns: 
            The object returned by the Scipy Optimizer. Contains information about the minimum, parameters, and a few other things
        """
        initialGuess = self.initialGuess
        if self.optimizerMethod == 'BFGS':
            optimiumReturn = scipy.optimize.minimize(self.CostFunction,initialGuess, method='BFGS', jac = self.gradCostFunction,options={'disp':True, 'gtol':self.accuracy, 'maxiter':self.steps})
        elif self.optimizerMethod == 'Powell':
            optimiumReturn = scipy.optimize.minimize(self.CostFunction,initialGuess, method='Powell',options={'disp':True, 'ftol':self.accuracy, 'maxiter':self.steps})
        self.kCoefs = optimiumReturn.x
        return optimiumReturn

    def generalCostFunction(self, thetas1, thetas2, index):
        '''
        This returns Tr(e<sup>i•thetas1•k1</sup>•v•e<sup>-thetas2•k2</sup>•H)
        (To make it clear, for Earp and Pachos function, we have k1=k2=k, thetas1=thetas2=thetas)

        Args:
            thetas1 (List): The coefficients for the k_tuples on the left side
            thetas2 (List): The coefficients for the K_tuples on the right side
            index (int): The number of k elements acting on v

        Returns:
            Tr(e<sup>i•thetas1•k1</sup>)•v•e<sup>-thetas2•k2</sup>•H)
        
        TODO: 
            * Comment the steps in this section
        '''
        if self.useCommTables:
            maxsize = 0
            #Prepares two lists: One for v matching dense coefficients with indices in g, One for H matching coefficients to indeices
            resV = [self.vcoefs,self.hElementIndices]
            resH = [self.hamiltonian.HCoefs, self.HElementIndices]

            for i in range(len(thetas1)-1,index,-1):
                resV = self.adj_action(thetas1[i],self.kElementIndices[i],resV[0],resV[1])

            
            #add each exp(thetas2*k2) to the list in reverse order and negative coefficients
            for i in range(index):
                resH = self.adj_action(-thetas2[i],self.kElementIndices[i],resH[0],resH[1])

            #create identity matrix for this dimensions
            I = (0,)*self.hamiltonian.sites
            
            
            resV = self.multiplyLinCombRound([1],[I],resV[0],resV[1],self.accuracy)  
            
            if index >= 0:
                resV = self.multiplyLinCombRound([math.cos(thetas1[index]),1j*math.sin(thetas1[index])],[I,self.cartan.k[index]],resV[0],resV[1],self.accuracy)    

                resV = self.multiplyLinCombRound(resV[0],resV[1],[math.cos(thetas2[index]),-1j*math.sin(thetas2[index])],[I,self.cartan.k[index]],self.accuracy)    

                    
            #get trace of v*H
            trace = 0
            for i in range(len(resV[0])):
                for j in range(len(resH[0])):
                    if resV[1][i] == self.cartan.g[int(resH[1][j])]: #Checks if the elements are the same
                        trace = trace + resV[0][i]*resH[0][j]
            
            return trace
        else: #Does not use the list index format, so computing commutators is slower
            maxsize = 0
            #Prepares two lists: One for v matching dense coefficients with indices in g, One for H matching coefficients to indeices
            resV = [self.vcoefs,self.cartan.h]
            resH = [self.hamiltonian.HCoefs, self.hamiltonian.HTuples]

            for i in range(len(thetas1)-1,index,-1):
                resV = self.adj_action(thetas1[i],self.cartan.k[i],resV[0],resV[1])

            
            #add each exp(thetas2*k2) to the list in reverse order and negative coefficients
            for i in range(index):
                resH = self.adj_action(-thetas2[i],self.cartan.k[i],resH[0],resH[1])

            #create identity matrix for this dimensions
            I = (0,)*self.hamiltonian.sites
            
            
            resV = self.multiplyLinCombRound([1],[I],resV[0],resV[1],self.accuracy)  
            
            if index >= 0:
                resV = self.multiplyLinCombRound([math.cos(thetas1[index]),1j*math.sin(thetas1[index])],[I,self.cartan.k[index]],resV[0],resV[1],self.accuracy)    

                resV = self.multiplyLinCombRound(resV[0],resV[1],[math.cos(thetas2[index]),-1j*math.sin(thetas2[index])],[I,self.cartan.k[index]],self.accuracy)    

                    
            #get trace of v*H
            trace = 0
            for i in range(len(resV[0])):
                for j in range(len(resH[0])):
                    if resV[1][i] == resH[1][j]:
                        trace = trace + resV[0][i]*resH[0][j]
            
            return trace
    def CostFunction(self, thetas):
        '''
        returns Tr(exp(thetas•k)•v•exp(-thetas•k)•H)
        '''
        
        val = self.generalCostFunction(thetas, thetas, -1)
        result = val.real

        return result

    def gradCostFunction(self, thetas):
        '''
        returns gradient of funky. Order of derivatives is the order of the parameters thetas.
        '''
        
        res = np.zeros(len(thetas))
        
        for i in range(len(thetas)):
            thetascopy = thetas.copy()
            thetascopy[i] = thetascopy[i]+math.pi/2
            
            diff = self.generalCostFunction(thetascopy,thetas, i) + self.generalCostFunction(thetas,thetascopy,i)
            
            res[i] = diff.real
            
        return res

    def adj_action(self,theta,k,coefs,tuples): 
        '''Computes Ad<sub>k</sub>(m) = e<sup>i•theta</sup>(coefs • tuples)e<sup>-i•theta</sup>
        Also known as the Adjoint Representation

        Args:
            theta (float): A single value, the coeffient of the k PauilString
            k (Tuple): A (PauliString)
            coefs (List of floats, can be complex): The coefficients indexed in order of the elements in tuples
            tuples (List of tuples): A List of (PauliStrings)
        
        Returns:
            The Algebraic element in m which is the result of the Adjoint Action (Representation)
        '''
        if self.useCommTables:
            result = [[],[]]
            
            for i in range(len(coefs)): #Generally, this is applied to v, an element in h and in m. Iterates over the elements in m
                m = tuples[i] #m is the index of some m element
                c = coefs[i] #c is the associate index
            
                res = [[c],[m]] #Just establishes the normal coefficient + tupleIndex pair
                comm = self.commutatePauliString(1,k,c/2,m) #Computes the commutator of the m currently in the loopand input k element,
                if comm[0] != 0:
                    res = [[c*math.cos(2*theta),1j*math.sin(2*theta)*comm[0]],[m,comm[1]]]  #If they do not commute, computes ?? (The exponential?)
                    
                for q in range(len(res[0])):
                    flag = 0
                    for j in range(len(result[0])):
                        if res[1][q] == result[1][j]:
                            result[0][j] = result[0][j] + res[0][q]
                            flag = 1
                            break
                    if flag == 0:
                        result[0].append(res[0][q])
                        result[1].append(res[1][q])
                            
            return result
        else:

            result = [[],[]]
            
            for i in range(len(coefs)): #Generally, this is applied to v, an element in h and in m. Iterates over the elements in m
                m = tuples[i] #m is the index of some m element
                c = coefs[i] #c is the associate index
            
                res = [[c],[m]] #Just establishes the normal coefficient + tupleIndex pair
                comm = self.commutatePauliString(1,k,c/2,m) #Computes the commutator of the m currently in the loopand input k element,
                if comm[0] != 0:
                    res = [[c*math.cos(2*theta),1j*math.sin(2*theta)*comm[0]],[m,comm[1]]]  #If they do not commute, computes ?? (The exponential?)
                    
                for q in range(len(res[0])):
                    flag = 0
                    for j in range(len(result[0])):
                        if res[1][q] == result[1][j]:
                            result[0][j] = result[0][j] + res[0][q]
                            flag = 1
                            break
                    if flag == 0:
                        result[0].append(res[0][q])
                        result[1].append(res[1][q])
                            
            return result
        
    def sethVecFromk(self):
        '''
        Returns h = exp(-thetas•k)•H•exp(thetas•k)

        Defines hErrorTuples and hErrorCoefs, which are the exact result of the adjoint representation. The result is in m, not in h, though it is mostly in h. 

        hCoefs and hTuples are the results stripped of the components in m
        '''

        I = (0,)*self.hamiltonian.sites
        
        res = [self.hamiltonian.HCoefs, self.hamiltonian.HTuples]
        
        #calculate exp(+...)*H*exp(-...)
        for i in range(len(self.kCoefs)):
            res = self.adj_action(-self.kCoefs[i],self.cartan.k[i],res[0],res[1])
        #Strips off error terms that ended up in m

        self.hErrorTuples = res[1]
        self.hErrorCoefs = res[0]

        self.hCoefs = []
        for w in self.cartan.h:
            hflag = 0
            for u in range(len(self.hErrorTuples)):
                if w == self.hErrorTuples[u]:
                    self.hCoefs.append(self.hErrorCoefs[u])
                    hflag = 1
                    break
            if hflag == 0:
                self.hCoefs.append(0)
        self.hTuples = self.cartan.h
        
    def errorhVec(self):
        '''Gets the norm square of the part in hcoefs•htuples that is orthogonal to Cartan subalgebra h.
        '''
        result = 0
        
        for i in range(len(self.hErrorCoefs)):
            
            term = self.hErrorTuples[i]
            flag = 0
            for j in range(self.lenh):
                if term == self.hTuples[j]:
                    flag = 1
                    break
            if flag == 0:
                result = result + abs(self.hErrorCoefs[i])**2
                
        return result

    def multiplyLinComb(self,A,tuplesA,B,tuplesB):
        '''
        Returns multiplication of two linear combinations of Pauli terms 
        '''

        a = len(A)
        b = len(B)
        
        C = []
        tuplesC = []
        csize = 0
        
        for i in range(a):
            for j in range(b):
                term = self.multiplyPauliString(A[i],tuplesA[i],B[j],tuplesB[j])
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
    
    def multiplyPauliString(self,a,tupleA,b,tupleB):
        """Computes the multiplication of two Pauli Strings representated as a tuple

        Args:
            a (np.complex128): 
                The coefficient of the first Pauli String term
            tupleA (Tuple or index in g):
                tuple represenation of the first Pauli String
            b (np.complex128):
                The coefficient of the second Pauli String term
            tupleB (Tuple or index in g): 
                tuple represenation of the second Pauli String

        Returns:
            c (np.complex128):
                The coefficient of the result a•TupleA . b•TupleB = c•TupleC, where c (the sign of the product of Paulis • a • b)
            tupleC (tuple) :
                the elementwise product of the PauliString, ignoring coefficients. 
        """
        if type(tupleA) != tuple:
            tupleA = self.cartan.g[int(tupleA)]
        if type(tupleB) != tuple:
            tupleB = self.cartan.g[int(tupleB)]
            
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
    
    def commutatePauliString(self,a,tupleA,b,tupleB):
        """Computes the commutator of two Pauli Strings representated as a tuple

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
                The coefficient of the result [a•TupleA,b•TupleB] = c•TupleC, where c is the Structure Constant • a • b.
            
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
            return (a*b*self.comm_coefs[int(tupleA)][int(tupleB)]), self.comm_table[int(tupleA)][int(tupleB)] 
    
    def multiplyLinCombRound(self,A,tuplesA,B,tuplesB, accur):
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
                term = self.multiplyPauliString(A[i],tuplesA[i],B[j],tuplesB[j])
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

    def printResult(self):
        """ Prints out the results of the otpimization """
        print("Printing Results:")
        print("K elements \n")
        for (co, tup) in zip(self.kCoefs, self.cartan.k):
            print(str(co).ljust(20) + '*' + str(IO.paulilabel(tup)))
        print('\n h elements: \n ')
        for (co, tup) in zip(self.hCoefs, self.cartan.h):
            print(str(co).ljust(25) + '*' + str(IO.paulilabel(tup)))
        print('Normed Error |KHK - Exact|:')
        U_cartan = CQS.util.verification.KHK(self.kCoefs, self.hCoefs,self.cartan.k, self.cartan.h)

        U_exact = CQS.util.verification.exactU(self.hamiltonian.HCoefs, self.hamiltonian.HTuples, 1)

        print(np.linalg.norm(U_exact - U_cartan))
    
    def saveKH(self):
        """
        Saves the information about the otimization to a .csv file
        """
        #Format: 
        #'h '              'hCoefs'           'k'        'kCoefs'
        # Tuple              Float           Tuple        Float
        fileNameCSV = self.saveFileName + '.csv'
        #Converts the formatting
        solutionList = [['h', 'hCoefs', 'k', 'kCoefs']]
        for (h, hCo, k, kCoefs) in zip(self.hTuples, self.hCoefs, self.cartan.k, self.kCoefs):
            solutionList.append([h, hCo, k, kCoefs])
        with open( fileNameCSV, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerows(solutionList)
        f.close()
        