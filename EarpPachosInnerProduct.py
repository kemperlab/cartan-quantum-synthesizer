# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 14:45:19 2020
Method to minimize in order to find the coefficients in the Cartan Decomposition

Two Sections:
    EarpPachosInnerProduct: Find a local minimum of <v, Ad_K(H)>, where v is a dense vector in h
@author: Efekan Kokcu
"""

""" ADJOINT INNER PRODUCT METHODS """
    

  
"""
This returns Tr(exp(thetas1*k1)*v*exp(-thetas2*k2)*H)
(To make it clear, for Earp and Pachos function, we have k1=k2=k, thetas1=thetas2=thetas)

Args:
    thetas1: List of Floats
        The coefficients for the first k vector
    k1: List of Tuples
        The PauliString basis for k1
    thetas2: List of Floats
        The coefficients for the second k vector
    k2: List of Tuples
        The PauliString basis for k2 - Often the same as k1
    h: List of Tuples
        The Pauli String basis for h
    Hcoefs: List of Floats
        The Hamiltonian Coefficients
    Htuples: List of Tuples
        The Hamiltonian elements as Pauli Strings

Return:
    res: float
        Trace to be minimized
"""

def generalInnerProduct(thetas1, k1, thetas2, k2, h, Hcoefs, Htuples):
   
    
    global callcount
    callcount = callcount + 1
    
    pi = math.pi
    hcoefs = [1]
    
    #generate v, a dense vector in the h algebra
    for i in range(len(h)-1):
        term = pi*hcoefs[i]
        while term > 1:
            term = term-1
        hcoefs.append(term)
      

    resH = [Hcoefs,Htuples] #
    resV = [hcoefs,range(len(k_tuples),len(k_tuples)+len(h_tuples))]
    
    maxsize = 0
    
    for i in range(len(thetas1)-1,index,-1):
        resV = adj_action(thetas1[i],k_ints[i],resV[0],resV[1])
        #if len(resV[0])>maxsize:
        #    maxsize = len(resV[0])
            
    #print(str(maxsize) + ' ' + str(len(k1)))
    
    maxsize = 0
    
    #add each exp(thetas2*k2) to the list in reverse order and negative coefficients
    for i in range(index):
        resH = adj_action(-thetas2[i],k_ints[i],resH[0],resH[1])
        #if len(resH[0])>maxsize:
        #    maxsize = len(resH[0])
                                  
    #print(str(maxsize) + ' ' + str(len(k1)))
    

    #create identity matrix for this dimensions
    I = (0,)*len(k[0])
    
    
    resV = multiplyLinCombRound([1],[I],resV[0],resV[1],accur)  
    
    if index >= 0:
        resV = multiplyLinCombRound([math.cos(thetas1[index]),1j*math.sin(thetas1[index])],[I,k[index]],resV[0],resV[1],accur)    
        resV = multiplyLinCombRound(resV[0],resV[1],[math.cos(thetas2[index]),-1j*math.sin(thetas2[index])],[I,k[index]],accur)    
    
            
    #get trace of v*H
    trace = 0
    for i in range(len(resV[0])):
        for j in range(len(resH[0])):
            if resV[1][i] == g_tuples[int(resH[1][j])]:
                trace = trace + resV[0][i]*resH[0][j]
    
    return trace


"""
Function of optimize over
Args:
    thetas: input to the optimizer
    k: Fully decomposed k'h'k', etc.
    h: First order Cartan Subalgebra of the Hamiltonian  algebra
    Hcoefs: Coupling
    Htuples: List of Pauli Tuples for the Hamiltonian
returns Tr(exp(thetas*k)*v*exp(-thetas*k)*H), which we want to minimize
"""
def adjointInnerProduct(thetas, *args):
    k, h, Hcoefs, Htuples = args
    return generalInnerProduct(thetas, k, thetas, k, h, Hcoefs, Htuples)



"""
returns gradient of funky. Order of derivatives is the order of the parameters thetas.
    List of partial derivatives for each theta, in the order of the input thetas
"""
def gradAdjointInnerProduct(thetas, k, h, Hcoefs, Htuples):
    
    res = []
    for i in range(len(thetas)):
        thetascopy = thetas.copy()
        thetascopy[i] = thetascopy[i]+math.pi/2
        
        diff = generalInnerProduct(thetascopy, k ,thetas, k, h, Hcoefs, Htuples) + generalInnerProduct(thetas, k ,thetascopy, k, h, Hcoefs, Htuples)
        
        res.append(diff)
    return res
        
    
"""
Returns h = exp(-thetas*k)*H*exp(thetas*k)
"""
def gethVecFromk(thetas, k, Hcoefs, Htuples):
    
    multiplylist = []
    I = (0,)*len(k[0])
    
    
    for i in range(len(thetas)):
        multiplylist.append([[math.cos(thetas[len(thetas)-i-1]),-1j*math.sin(thetas[len(thetas)-i-1])],[I,k[len(thetas)-i-1]]])
    
    multiplylist.append([Hcoefs,Htuples])

    for i in range(len(thetas)):
        multiplylist.append([[math.cos(thetas[i]),1j*math.sin(thetas[i])],[I,k[i]]])
    
    while len(multiplylist) > 1:
        for i in range(int(len(multiplylist)/2)):
            term = multiplyLinComb(multiplylist[i][0],multiplylist[i][1],multiplylist[i+1][0],multiplylist[i+1][1])
            multiplylist[i][0] = term[0]
            multiplylist[i][1] = term[1]
            multiplylist.pop(i+1)
        
    hcoefs = multiplylist[0][0]
    htuples = multiplylist[0][1]

    return hcoefs, htuples
