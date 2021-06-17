"""
Class contains information about the Hamiltonian to be decomposed.

##Functionality:
* Build Hamiltonians from prebuilt Spin models
* Add custom Hamiltonians 

TODO:
 * Add Hubbard Model

Authors:
* Thomas Steckmann
* Efekan Kokcu
"""
import warning as warn
class Hamiltonian:
    def __init__(self, sites, name = None):
        """Initializes an emtpy Hamiltnoan, unless name is specified

        Args:
            qubits (int): The number of lattice points the Hamiltonisn svyd on
            name (List of Tuples): Easily build Hamiltonians from native models. 

                ### Options:
                    * `[(coefficient, 'modelname', periodic boundary condisitons boolian)]` - Builds a single Hamiltonian with constant coefficient. Periodic boundary conditions are true or false, false is assumed if not specified
                    * `[([list of coefficients], 'modelname')]` - Populates the Hamiltonian using the list of coefficients. Lengths must match
                    * `[(coefficient, 'modelname1'),([list of coefficients], 'modelname2')]` - Combines two Hamiltonains. Order my alter default choice of 𝖍
                #### Examples:
                    * `[(1,'XY', true)]` on three qubits gives: XXI + YYI + IXX + IYY + XIX + YIY
                    * `[([1,2],'kitaevEven')] on three qubits gives: 1*XXI + 2*IYY 
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
                    - hubbard: (XXII + YYII + IIXX + IIYY + ZIZI + IZIZ)
        """
        self.sites = sites
        self.hamiltonian = []
        #Builds the Hamiltonian
        if name is not None:
            for pair in name: 
                addModel(pair)

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
                - hubbard: (XXII + YYII + IIXX + IIYY + ZIZI + IZIZ)
    """
    if len(pair == 3): #Handles the boundary condition variable format
        hamiltonianTerms = generateHamiltonian(pair[1], pair[2]) #pair[1] = modelname, pair[2] = boundary condition
        if isinstance(pair[0], list): #pair[0] = coefficients float or list
            try:
                pairedIterable = zip(pair[0],hamiltonianTerms)
            except:
                warn.warn("Coefficient List mismatch")
            for (coefficient, pauliString) in pairedIterable:
                self.hamiltonian.append((coefficient, pauliString))
        elif pair[0]:
            coefficient = pair[0]
            for pauliString in hamiltonianTerms:
                self.hamiltonian.append((pair[0], hamiltonianTerms))

def addTerms(self, pairs):
    """Adds custom elements to build out the Hamiltonian

    Args:
        pair (tuple): Formatted as either
            * `[(coefficient, PauliString Tuple), (coefficient, PauliStringTuple),...]`
            * `(coefficient, PauliString Tuple)`

    Examples:
        * `Hamiltonian.addTerms((0.45, (1,3,2,0,0,0)))`
        * `Hamiltonian.addTerms([(1,(1,1,0,0)),(2,(2,2,0,0))])

    """
    if isinstance(pairs, list):
        for pair in pairs:
            self.hamiltonian.append((pair))
    elif isinstance(pairs, tuple):
        self.hamiltonian.append(pairs)





def generateHamiltonian(self, modelType, closed = False):
    """Helper Function to generate Hamiltonians.
    
    From the model type, generates the Hamiltonian Pauli Strings
    
    For valid models, see 

    Args: 
        modelType (Tuple of Strings): 
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
            term = ((0,) * i) + (1,) + ((0,) * (self.sites - 1 - i))
            hamiltonian.append(term)

    else:
        raise Exception("Invalid Model. Valid models include: 'XX', 'YY', 'ZZ', 'XY', 'KitaevEven', 'KitaevOdd', 'Heisenberg', TransverseIsing', and 'Transverse_Z'")
    return hamiltonian
