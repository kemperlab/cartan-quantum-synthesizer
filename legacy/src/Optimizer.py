# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:14:40 2020

General Methods to compute the exponential coefficients in the Cartan Decomposition of a 1D spin lattice hamiltonian

Depends on:
    - MinimizerFunctions.py
        - adjointRepNorm
        - adjointAction
        - ajointInnerProduct
        - generalInnerProduct
        - gradAdjointInnerProduct
        - gethVecFromk
        
    - HamiltonianAlgebra.py
        - generateAlgebra
        - hamiltonianTuple
        - 
        
    - CartanMethods.py
        - MakeGroup (Hamiltonian)
        - findh(m)
        - getSubalgebra(m)
        - getSubalgebraElem(m,element)
        
    - PauliOps.py
        - commutePauliString
        - multiplyPauliString
        - multiplyLinComb
        - simplifyLinComb
        
    - IO.py
        - saveCSV
        - printList

@author: Thomas Steckmann
@author: Efekan Kokcu
"""

#Import General Python Packages
import numpy as np
from warnings import warn
import time
import csv
#import os.path
import scipy.optimize
#import matplotlib.pyplot as plt
import random as rand

#Import Package Elements
from src.MinimizerFunctions import adjointRepNorm
from src.HamiltonianAlgebra import generateAlgebra, hamiltonianTuples
#from EarpPachosInnerProduct import
#from CartanMethods import makeGroup, findh, getSubalgebra, getSubalgebraElem
#from PauliOps import SIGN_RULES, X, Y, Z, I, ops, paulis, commutePauliString, multiplyPauliString, MultiplyLinComb, simplifyLinCom
#from IO import saveCSV, printList


def FindParameters(modelType, J, sites, closed, fileName=None ,depth = 10, acc = 1e-5, options =None):
    """Hopefully the only function you will need to use. 

    Takes all the parameters of your model and output options and produces the Gates and gate angles

    Args:
        ModelType (tuple of Strings)
            Valid models include: 
                'X',
                'Y',
                'Z',
                'XY',
                'XY2D',
                'KitaevEven',
                'KitaevOdd',
                'Heisenberg',
                TransverseIsing',
                'Transverse_Z'
            Models can be combined to get different coupling, such as Jx = 0.5 and Jy= 1 being described as ('X','Y')
        J (tuple of Floats OR Lists of Floats):
            Depending on the model type, J is either a tuple for coupling corresponding to the same index model,
            or it is a tuple of tuples with one coupling for each intereaction in the Hamiltonian. Each J is shared by site 
            for the specificed Hamiltonian                              
        closed (bool):
            True if Periodic Boundary conditions
        fileName (String):                                                                           
            Determines the read/write file location, if such behavior is desired
        sites (int):
            3 - 8 (No hard maximum, just runtime)
        depth (int): 10
            The number of terms in the BCH lemma expansion of Adjoint Representation
            Not needed for EarpPachosInnerProduct
        acc (float):
            Determines the convergence criteria
        options (dict)::
            {
            'evolve': string                                                                          #######Not yet implemented###########
                Choose the path of the optimizer. Default is to use the guess of 1 for all h elements and 0 for all k elements
                Evolving the Hamiltonian to turn on coupling might be faster or might produce more consistent results. 
                options:
                    None
                    'nextTerm'
                    'split',
            'ordered': True,
            'return': Tuple of strings
                Determines the output method. Options include:
                    'python'
                    'plot'
                    'mathemtica'                                                                                    
                    'csv',
            'debug': bool
                Print details about the optimization, such as the hamiltonian, algebra, and convergence,        
            'steps': int
                The Maximum number of iterations or steps in the optimizer,
            'statusReporter': bool
                If True, the program will print progress messages to console
            'loadFile': bool                                                                       
            'method': string                                                                 
                Determines the method of minimization
                Options include
                    'AdjointRep'
                    'AdjointInnerProduct',
            'optimizerFunction': string
                The minimizer function to be used. Defaults to Powell for AdjointRep, BFGS for AdjointInnerProduct. Easy to add more options
                Options:
                    'Powell'
                    'BFGS',
            }

    Todo:
        * Fix Status Reporter
        * File Loading
        * Evolve Optimization Path (Turn on coupling slowly)
        * File Name Behavior
        * Cartan Subalgebra Choices
        * EarpPachosInnerProduct Methods
            
    """


    """Checking input"""
    #Input Checking for a few confusing methods
    if not isinstance(modelType, tuple):
        raise Exception('modelType must be a Tuple')
    if not isinstance(J, tuple):
        raise Exception('J must be a Tuple')
    if not (isinstance(options,dict) or options is None):
        raise Exception('Options must be a dictionary or None Type')
    
    
    ########################################
    """Initialize the options dictionary"""
    ########################################

    
    OPTIONS = {'evolve': None, #Not yet implemented
               'ordered': True,
               'return': ('python',),
               'debug': False,
               'steps': 5000,
               'statusReporter': False,
               'loadFile': False, #Not yet implemented
               'method': 'AdjointInnerProduct', #Not yet implemented
               'optimizerFunction': 'Powell' #Powell will always work
               }
    if OPTIONS['method'] == 'AdjointInnerProduct': #Best for AdjointInnerProduct is BFGS b/c it uses gradient
        OPTIONS['optimizerFunction'] = 'BFGS'
    
    #If user input options are provided, overwrite the default options
    if options is None:
        pass
    else:
        for key in dict(options): #copies the user specific options to the initialized dictionary
            OPTIONS[key] = options[key]
    

    ########################################
    """Specifies the user specified output"""
    ########################################


    #Intialize Dict of possible methods of returning the solution, initialized to False all except the python output
    returnMethod = {"python": False, 'plot': False, 'mathematica':False, 'csv': False}
    
    #Scans the user input options for return and updates the initialized dict. Also verifies the user input
    for key in OPTIONS['return']: 
        if key == 'csv': #Verifies if the fileName is writable and exists if the user wants to save the file
            if fileName == None:
                raise Exception("SaveToCSV requires a valid fileName")
        fileNameCSV = fileName + '.csv'
        #Test Write
        with open(fileNameCSV, "w") as f:
                writer = csv.writer(f)
                writer.writerows('')
        f.close()
        if key == 'loadFile': #Warns the user that loadFile does not do anything
            warn("File Loading has not yet been implemented", RuntimeWarning)
        returnMethod[key] = True
        
    #If the model is disordered, the user needs to list the desired coupling constants.  This checks that the inputs are correct
    if not OPTIONS['ordered']:
        for parameters, model in zip(J,modelType):
            if not isinstance(parameters, tuple):
                raise Exception('Disordered systems require Coupling constants listed in a tuple with length appropriate for the number of sites')
            if model == "TransverseIsing":
                raise Exception('For disordered transverse Ising, please list the X and Z terms seperately as modelType =("X","Transverse_Z")')
            elif model == "Transverse_Z" and len(parameters) != (sites):
                raise Exception('Disordered systems require Coupling constants listed in a tuple with length appropriate for the number of sites')
            elif len(parameters) != (sites - 1 + closed):
                raise Exception('Disordered systems require Coupling constants listed in a tuple with length appropriate for the number of sites')
    else:
        for parameters in J:
            if isinstance(parameters, tuple):
                raise Exception('Ordered system coupling must be specificed for a single Float coupling constant shared by all sites')


    #initialized veriables from the OPTIONS dict
    loadFile = OPTIONS['loadFile']
    global statusReporter  #Temporary status reporter options. TODO: Allows a count of how many functions have run
    statusReporter = int(OPTIONS['statusReporter'])
    debug = OPTIONS['debug']
    steps = OPTIONS['steps']
    ordered = OPTIONS['ordered']
    method = OPTIONS['method']
    optimizerFunction = OPTIONS['optimizerFunction']


    ########################################
    """Optimizer Implementation - Special cases, algebra generation, and optimizer selection """
    ########################################
    
    
    """ Special Cases """
    #There is a special case for the XY model, in which we can use two Kitaev Models instead of one XY model.
    replaceKitaev = False
    if "XY" in modelType and (sites % 2 == 0 or not closed) and ordered: #This can be modified later to allow for disordered systems specified by one list of coupling (shared Jx and Jy at each site)
        replaceKitaev = True
    
    

    
    """ Algebra Generation"""
    #Generates the populated coefficient Hamiltonian in the m algebra basis, finds the hm algebra (h terms followed by m terms)
    #, the k and the h algebras using a function call. See documentation in GenerateHamiltonian.py
    Hcoeff = None
    if replaceKitaev: #Instead of running 1 XY, run 1 Kitaev, and use that result to generate the Kitaev result (saves time)
        if method =='AdjointInnerProduct':
            H, hm, k, h  = generateAlgebra(sites, ('KitaevEven',), J, closed)
            HTuples = hamiltonianTuples(sites, 'KitaevEven', closed)
            Hcoeff = np.ones(len(HTuples)) * J[0]
        else:
            H, hm, k, h  = generateAlgebra(sites, ('KitaevEven',), J, closed)
    elif method =='AdjointInnerProduct':
        H, hm, k, h  = generateAlgebra(sites, modelType, J, closed) 
        HTuples = hamiltonianTuples(sites, modelType, closed)
        Hcoeff = np.ones(len(HTuples)) * J[0]
    else:
        H, hm, k, h  = generateAlgebra(sites, modelType, J, closed) 

    
    #Reports on the size of the optimization, the parameters, and the algebras if the user specifies
    if statusReporter:
        if method == ['AdjointInnerProduct']:
            print("Beginning minimization over %s parameters" % len(k))
        else:
            print("Beginning optimization over %s parameters" % len(hm))
    if debug:
        print('h = ')
        print(h)
        print('hm = ')
        print(hm)
        print('k = ')
        print(k)
    
    
    #Intial Guess: 1 for all h terms, 0 for all k Terms
    if method == 'AdjointRep':
        initialGuess = np.concatenate((np.ones(len(h)), np.zeros(len(k))))      
    elif method == 'AdjointInnerProduct':
        initialGuess = np.zeros(len(k))
    else:
        initialGuess = None
    
    ################################################################################################
    """The Optimizer. Defined internally to allow for easy customization based on the method options
       https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html"""
    ################################################################################################
    def optimize(initialGuess):
        start_time = time.time()
        #Passes the Hamiltonian through the args, calle the "function" to minimize using the initial Guess we pass it initially
        """Toggle options for the minimizer"""
        #Currently, we only use the Powell method. In the future, we can add an option to change method for minimization
        if method == 'adjointAction':
            optimiumReturn = scipy.optimize.minimize(adjointRepNorm,initialGuess, args = (depth, H, hm, k, h), method=optimizerFunction,options={'disp':(not statusReporter == 0), 'ftol':acc, 'maxiter':steps})# "xtol": accuracy})#, 'gtol':1e-20 ,'maxiter':500})
        elif method == 'AdjointInnerProduct':
            pass

            ########################################################
            ########################################################
            #Efekan Code Goes Here
            ########################################################
            ########################################################

        else:
            optimiumReturn = None
            raise Exception('Incorrect Method')
        print("--- %s seconds ---" % (time.time() - start_time))
        
        return optimiumReturn
    
    #runs the optimizer functions
    result = optimize(initialGuess)
    x = result.x #See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
    print(x)
    
    print('Printing E&P k results')
    if method == 'AdjointInnerProduct':
        print([b for b in zip(x, k)])
    
    
    #Special Cases combining two Kitaev Models into one XY model
    if replaceKitaev:
        #evenAlgebra
        H0, hm0, k0, h0  = generateAlgebra(sites, ('KitaevEven',), J, closed)
        #oddAlgebra 
        H1, hm1, k1, h1  = generateAlgebra(sites, ('KitaevOdd',), J, closed)
        #The k coefficients are negated, while the h coefficients are shared
        xh0 = x[:-len(k)]
        xk0 = x[len(h):]
        xh1 =  1 * xh0
        xk1 = -1 * xk0
        #full solution
        h = h0 + h1
        k = k0 + k1
        hm = hm0 + hm1

        x = np.concatenate((xh0 , xh1 , xk0 , xk1))
        H, a, b, c = generateAlgebra(sites, modelType, J, closed)
        #The algebra terms are now returned to normal and we can verify the solution

    ########################################################################
    """OUTPUT processing and saving"""
    ########################################################################

    #Always prints the result
    print(hm)
    print(x)
    if returnMethod['python']:
        print([a for a in zip(x[:len(h)], h)])
        print([b for b in zip(x[ len(h):], k)])


    #Selects which outputs to run
    if returnMethod['mathematica']:    
        pass
        #mathematicaString = buildMathematicaResult(results,hm, h, k)
    if returnMethod['csv']:
        #Saves as CSV
        #Format: 
        #'h and k algebra' 'coefficients' 'hm algebra' 'hamiltonian'
        # Tuple              Float           Tuple        Float
        fileNameCSV = fileName + '.csv'
        #Converts the formatting
        solutionList = [['h and k algebra', 'coefficients', 'hm algebra' ,'hamiltonian']]
        for hkAlgebra, coefficient, hmAlgebra, hamiltonian in zip(h + k,x,hm, H):
            solutionList.append([hkAlgebra, coefficient, hmAlgebra, hamiltonian])
        with open( fileNameCSV, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerows(solutionList)
        f.close()
    return x[:-len(k)] , h, x[ len(h):], k



#sites = 4
#ID = 'ML_Data\\MLdata_' + '4_XY_Open_OldMethod_file1'
#FindParameters(('XY',),(1,),sites,False, fileName= ID ,options = {'statusReporter': True, 'debug': True, 'return': ('python', 'csv'), 'method': 'adjointAction'})
#print(ID)
