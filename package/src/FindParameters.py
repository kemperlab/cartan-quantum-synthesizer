# -*- coding: utf-8 -*-
__docformat__ = 'google'
""" From a Cartan Decomposition, runs the optimizer to find the appropriate parameters for a circuit

Warning: Very long runtime for more than 8ish qubits

Authors:
* Thomas Steckmann
* Efekan Kokcu
"""


import time
import scipy.optimize
import numpy as np
import csv
import os
import math
import package.util.IO as IO

#Commutator tables
RULES = [1,3,1,3]
SIGN_RULES = [[1,1,1,1], 
             [1, 1, 1j, -1j],
             [1, -1j, 1, 1j],
             [1, 1j, -1j, 1]]

class FindParameters:
    """
    Class to find, manage, and store information about the parameters needed in the decomposition
    """
    def __init__(self, cartan, saveFileName = None, loadfileName=None, optimizerMethod='BFGS', accuracy=1e-5, initialGuess=None, steps = 5000):
        """
        Initializing a FindParameters class automatically runs the optimizer over the Cartan decomposition and provided Hamiltonian
        
        If saveFileName is provided, verifies file location before proceeding, and saves the output as a csv
        If loadFileName is provided, does not run the optimizer and instead reads data from the provided file

        TODO:
            * Write Load file functions
        
        Args:
            cartan (Obj Cartan): The Cartan object containing the k,h, and Hamiltonian information
            saveFileName (String, default=None): path to save the output.
            loadFileName (String, default=None): path to a csv containing previous optimizer results
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
        self.hamiltonian = self.cartan.hamiltonian #Extracts the hamiltonain object
        self.optimizerMethod = optimizerMethod
        self.accuracy = accuracy
        self.lenK = len(self.cartan.k)
        self.lenh = len(self.cartan.h)
        self.steps = steps
        #Begin Optimizer
        if loadfileName is not None: #If able to, loads prior results
            raise Exception('Unable to continue, file loading is not implemented')
        else:   
            if saveFileName is not None:
                if not os.access(saveFileName, os.W_OK): #Verifies write permission
                    raise Exception('Save File location does not have write access. Aborting Optimization.')
            
            #Generating reused values before begining the optimizer loop
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
        I = (0,)*len(self.cartan.k[0])
        
        
        resV = self.multiplyLinCombRound([1],[I],resV[0],resV[1],self.accuracy)  
        
        if index >= 0:
            resV = self.multiplyLinCombRound([math.cos(thetas1[index]),1j*math.sin(thetas1[index])],[I,self.cartan.k[index]],resV[0],resV[1],self.accuracy)    

            resV = self.multiplyLinCombRound(resV[0],resV[1],[math.cos(thetas2[index]),-1j*math.sin(thetas2[index])],[I,self.cartan.k[index]],self.accuracy)    

                
        #get trace of v*H
        trace = 0
        for i in range(len(resV[0])):
            for j in range(len(resH[0])):
                if resV[1][i] == self.cartan.g[int(resH[1][j])]:
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

        I = (0,)*len(self.cartan.k[0])
        
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
        


    
