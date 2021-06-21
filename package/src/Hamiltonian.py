
# -*- coding: utf-8 -*-
__docformat__ = 'google'
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
from package.util.IO import printlist, paulilabel

#from ..package.IO import printlist, paulilabel

class Hamiltonian:
    def __init__(self, sites, name = None):
        """Initializes an emtpy Hamiltnoan, unless name is specified
        Args:
            qubits (int): The number of lattice points the Hamiltonisn svyd on
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
            hamiltonianTerms = self.generateHamiltonian(pair[1]) #pair[1] = modelname, pair[2] = boundary condition
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
                self.HTuples.append(pair[1])
                self.HCoefs.append(pair[0])
        elif isinstance(pairs, tuple): #Verifies user input - if a tuple, just adds it
            if isinstance(pairs[0], list):
                for i in pairs[0]:
                    self.HCoefs.append(i)
                for i in pairs[1]:
                    self.HTuples.append(i)
            else:
                self.HTuples.append(pairs[1])
                self.HCoefs.append(pairs[0])



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