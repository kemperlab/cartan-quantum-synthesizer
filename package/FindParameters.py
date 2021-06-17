# -*- coding: utf-8 -*-
__docformat__ = 'google'
""" From a Cartan Decomposition, runs the optimizer to find the appropriate parameters for a circuit

Warning: Very long runtime for more than 8ish qubits

TODO: 
    Fix the issues with commutatePauliString and the table - probably bring that function here
Authors:
* Thomas Steckmann
* Efekan Kokcu
"""

import IO
import time
import scipy.optimize
import numpy as np
import csv
import os
from PauliOps import commutatePauliString, multiplyLinCombRound
import math

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
                ### Options: 
                    * `'BFGS'` : Used Gradient
                    * '`Powel'`: Does not use Gradient
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
        elif saveFileName is not None: 
            if not os.access(saveFileName, os.W_OK): #Verifies write permission
                raise Exception('Save File location does not have write access. Aborting Optimization.')
            
            #Generating reused values before begining the optimizer loop
            self.setCommutatorTables() #Sets values for a look-up table for the commutators
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

    def setCommutatorTables(self):
        """ 
        Generates commutator look-up tables for the commutators of different elements
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
                res = commutatePauliString(1,g_tuples[i],1,g_tuples[j])
                
                self.comm_coefs[i][j] = res[0]
                
                if res[0]==0:
                    self.comm_table[i][j]=0
                else:
                    for q in range(len(g_tuples)):
                        if res[1] == g_tuples[q]:
                            self.comm_table[i][j] = q
                            
        #Computes the indices of each of the k elements in the g list                   
        self.kElementIndices = []
        for i in range(len(k_tuples)):
            for j in range(len(g_tuples)):
                if k_tuples[i] == g_tuples[j]:
                    self.kElementIndices.append(j)
                    break  

        #computes the indicies of each of the Ham elements in the g list
        self.HElementIndices = [] 
        Htuples = self.hamiltonian.getHamTuples()
        for i in range(len(Htuples)):
            for j in range(len(g_tuples)):
                if g_tuples[j]==Htuples[i]:
                    self.HElementIndices.append(j)
                    break

    
    def optimize(self):

        if self.optimizerMethod == 'BFGS':
            optimiumReturn = scipy.optimize.minimize(self.CostFunction,self.initialGuess, method='BFGS', jac = self.gradCostFunction,options={'disp':True, 'gtol':self.accuracy, 'maxiter':self.steps})
        elif self.optimizerMethod == 'Powell':
            optimiumReturn = scipy.optimize.minimize(self.CostFunction,self.initialGuess, method='Powell',options={'disp':True, 'ftol':self.accuracy, 'maxiter':self.steps})
        self.kCoefs = optimiumReturn
        return optimiumReturn


    def generalCostFunction(self, thetas1, thetas2, index):
        '''
        This returns Tr(exp(thetas1*k1)*v*exp(-thetas2*k2)*H)
        (To make it clear, for Earp and Pachos function, we have k1=k2=k, thetas1=thetas2=thetas)

        Args:
            thetas1 (List): The coefficients for the k_tuples on the left side
            thetas2 (List): The coefficients for the K_tuples on the right side
            index (int): The number of k elements acting on v

        Returns:
            Tr(exp(thetas1*k1)*v*exp(-thetas2*k2)*H)
        
        TODO: Comment the steps in this section
        '''

        maxsize = 0

        resV = [self.vcoefs,range(self.lenK,self.lenK+self.lenh)]
        resH = [self.hamiltonian.HCoefs, self.hamiltonian.HTuples]

        for i in range(len(thetas1)-1,index,-1):
            resV = self.adj_action(thetas1[i],self.kElementIndices[i],resV[0],resV[1])

        
        #add each exp(thetas2*k2) to the list in reverse order and negative coefficients
        for i in range(index):
            resH = self.adj_action(-thetas2[i],self.kElementIndices[i],resH[0],resH[1])

        #create identity matrix for this dimensions
        I = (0,)*len(self.cartan.k[0])
        
        
        resV = multiplyLinCombRound([1],[I],resV[0],resV[1],self.accuracy)  
        
        if index >= 0:
            resV = multiplyLinCombRound([math.cos(thetas1[index]),1j*math.sin(thetas1[index])],[I,self.cartan.k[index]],resV[0],resV[1],self.accuracy)    
            resV = multiplyLinCombRound(resV[0],resV[1],[math.cos(thetas2[index]),-1j*math.sin(thetas2[index])],[I,self.cartan.k[index]],self.accuracy)    
        
                
        #get trace of v*H
        trace = 0
        for i in range(len(resV[0])):
            for j in range(len(resH[0])):
                if resV[1][i] == self.cartan.g[int(resH[1][j])]:
                    trace = trace + resV[0][i]*resH[0][j]
        
        return trace
    
    def CostFunction(self, thetas):
        '''
        returns Tr(exp(thetas*k)*v*exp(-thetas*k)*H)
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

    def adj_action(theta,k,coefs,tuples): 
        '''Computes Ad_k(m) = e<sup>i*theta</sup>(coefs * tuples)e<sup>-i*theta</sup>
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
        
        for i in range(len(coefs)):
            m = tuples[i]
            c = coefs[i]
        
            res = [[c],[m]]
            comm = commutatePauliString(1,k,c/2,m)
            if comm[0] != 0:
                res = [[c*math.cos(2*theta),1j*math.sin(2*theta)*comm[0]],[m,comm[1]]]  
                
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
        Returns h = exp(-thetas*k)*H*exp(thetas*k)
        '''

        I = (0,)*len(self.cartan.k[0])
        
        res = [self.hamiltonian.HCoefs, self.hamiltonian.HTuples]
        
        #calculate exp(+...)*H*exp(-...)
        for i in range(len(self.kCoefs)):
            res = self.adj_action(-self.kCoefs[i],self.cartan.k[i],res[0],res[1])
            
        hcoefs = res[0]
        htuples = res[1]

        self.hCoefs
        self.hTuples
    
    def errorhVec(self):
        '''Gets the norm square of the part in hcoefs*htuples that is orthogonal to Cartan subalgebra h.
        '''
        result = 0
        
        for i in range(len(self.hCoefs)):
            
            term = self.hTuples[i]
            flag = 0
            for j in range(self.lenh):
                if term == self.cartan.h[j]:
                    flag = 1
                    break
            if flag == 0:
                result = result + abs(self.hCoefs[i])**2
                
        return result
    
    
