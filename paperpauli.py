import numpy as np
import math
import numpy.linalg as la
import scipy.linalg as scila
from numpy import kron
import time
import scipy.optimize
import matplotlib.pyplot as plt


ops = ['I','X','Y','Z']
"""
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
"""
RULES = [1,3,1,3]
#In version 3, these terms are corrected
"""
SIGN_RULES: 
    Gives the multiplication sign rules for multiplying Pauli Matricies (ex. X*Y -> iZ)
    
  I  X  Y  Z
I +  +  +  +
X +  +  +i -i
Y +  -i +  +i
Z +  +i -i +

Order: row * column
"""
SIGN_RULES = [[1,1,1,1], 
             [1, 1, 1j, -1j],
             [1, -1j, 1, 1j],
             [1, 1j, -1j, 1]]

#The Pauli Matricies in matrix form
X = np.array([[0,1],[1,0]])
Y = np.array([[0,-1j],[1j,0]])
Z = np.array([[1,0],[0,-1]])
I = np.array([[1,0],[0,1]])
#Allows for indexing the Pauli Arrays (Converting from tuple form (0,1,2,3) to string form IXYZ)
paulis = [I,X,Y,Z]


    
        
g_tuples = []              #g = k + h + (m-h)         
k_tuples = []
m_tuples = []
h_tuples = []
comm_table = []
com_coefs = []


'''
    Sets the global tuples up there and generates commutation table, so that we wouldn't have to calculate their commutation again and again.
'''
def set_tuples(k,m,h):
    
    global g_tuples
    global k_tuples
    global m_tuples
    global h_tuples
    
    k_tuples = k.copy()
    h_tuples = h.copy()
    m_tuples = m.copy()
    
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
    
    global comm_table
    global comm_coefs
    
    comm_table = np.zeros((len(g_tuples),len(g_tuples)))
    comm_coefs = np.zeros((len(g_tuples),len(g_tuples)),dtype=np.complex_)
    
    for i in range(len(g_tuples)):
        for j in range(len(g_tuples)):
            res = commutatePauliString(1,g_tuples[i],1,g_tuples[j])
            
            comm_coefs[i][j] = res[0]
            
            if res[0]==0:
                comm_table[i][j]=0
            else:
                for q in range(len(g_tuples)):
                    if res[1] == g_tuples[q]:
                        comm_table[i][j] = q
                        break
            


 
"""
Computes the commutator of two Pauli Strings representated as a tuple
Args:
    a: np.complex128 
        The coefficient of the first Pauli String term
    tupleA: Tuple 
        tuple represenation of the first Pauli String
    b:np.complex128 
        The coefficient of the second Pauli String term
    tupleB: Tuple 
        tuple represenation of the second Pauli String
Returns:
    c: np.complex128
        The coefficient of the result [a*TupleA,b*TupleB] = c*TupleC, where c is the Structure Constant * a * b
    tupleC: Tuple 
        the elementwise commutator of the PauliString, ignoring coefficients. 
"""
def commutatePauliString(a,tupleA,b,tupleB):
    
    if type(tupleA)!=int:
    
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
        '''
        print('In there\n')
        print(comm_table)
        print(comm_coefs)
        print(tupleA)
        print(tupleB)
        '''
        return (a*b*comm_coefs[int(tupleA)][int(tupleB)]), comm_table[int(tupleA)][int(tupleB)] 
    
    
"""
Computes the multiplication of two Pauli Strings representated as a tuple
Args:
    a: np.complex128 
        The coefficient of the first Pauli String term
    tupleA: Tuple 
        tuple represenation of the first Pauli String
    b:np.complex128 
        The coefficient of the second Pauli String term
    tupleB: Tuple 
        tuple represenation of the second Pauli String
Returns:
    c: np.complex128
        The coefficient of the result a*TupleA . b*TupleB = c*TupleC, where c (the sign of the product of Paulis * a * b)
    tupleC: Tuple 
        the elementwise product of the PauliString, ignoring coefficients. 
"""
def multiplyPauliString(a,tupleA,b,tupleB):
    if type(tupleA) != tuple:
        tupleA = g_tuples[int(tupleA)]
    if type(tupleB) != tuple:
        tupleB = g_tuples[int(tupleB)]
        
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


'''
    Returns multiplication of two linear combinations of Pauli terms, and rounds things that are smaller than accur to zero. 
'''

def multiplyLinCombRound(A,tuplesA,B,tuplesB, accur):

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


def commutateLinComb(A,tuplesA,B,tuplesB,accur):
    '''
    AB = multiplyLinCombRound(A,tuplesA,B,tuplesB, accur)
    BA = multiplyLinCombRound(B,tuplesB,A,tuplesA, accur)
    
    
    for i in range(len(BA[0])):
        BA[0][i] = -BA[0][i]
        
    
    coefs = AB[0] + BA[0]
    tuples = AB[1] + BA[1]
    
    simplifyLinComb(coefs,tuples)
    
    return coefs,tuples
    '''
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


'''
    Simplifies lin comb of Pauli matrices that it eats. Doens't return anything
'''
def simplifyLinComb(A,tuples):
    
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
            
            
def getmatrixterm(pterm):
    
    result = np.eye(1)
    for p in pterm:
        pmat = 0
        if p==0:
            pmat = I
        elif p==1:
            pmat = X
        elif p==2:
            pmat = Y
        elif p==3:
            pmat = Z
        
        result = np.kron(result,pmat)
        
    return result
    
def getmatrices(tuples):
    
    result = []
    for i in range(len(tuples)):
        result.append(getmatrixterm(tuples[i]))
    return result

def getmatrix(coefs,tuples):
    
    result = 0
    for i in range(len(coefs)):
        if i == 0:
            result = coefs[i]*getmatrixterm(tuples[i])
        else:
            result = result + coefs[i]*getmatrixterm(tuples[i])
        
    return result



'''
    Following function returns 0 if tuple m is not incu=luded in tuple list g, returns 1 if it is included.
'''
def included(g,m):
    L = len(g)
    res = 0    
    for i in range(L):
        if g[i]==m:
            res = 1
            break
    
    return res


    
'''
    Following function returns a closure of a given list of pauli strings (g). The list doesn't include any coefficients, it is just
    a tuple like (0,2,3) representing IYZ.
'''
def makeGroup(g):
    
    flag = 0
    while (flag == 0):
        flag = 1
        L = len(g)
        #initialize commutations
        coms = []

        #calculate all possible commutations and
        for i in range(L):
            for j in range(i,L):
                m = commutatePauliString(1,g[i],1,g[j])
                
                
                #add all new ones to the list
                if (abs(m[0])>0) & (included(coms,m[1])==0) & (included(g,m[1])==0):
                    #set flag to 0 whenever there is a new term to be added
                    flag = 0
                    coms.append(m[1])

        #then merge initial list with these new commutations
        g = g + coms
        #print(g)
    return g





'''
DECOMPOSITIONS
'''

'''
    Following one implements the even-odd decomposition i.e it counts the number of Pauli matrices in a Pauli term.
    It corresponds to \theta(g) = -YYY..Y g^T YYY..Y
'''
def evenodd(g):
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
            
    return k,m
 
    
'''
    Following one counts the number of given elements (X,Y or Z in number), and puts even numbers in m, odd numbers in k. 
    For element=2, it corresponds to \theta(g) = -g^T
'''    
def elemcount(g,element):
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
        
    return k,m


'''
    Following one is the Kneja Glaser decomposition. 
    It corresponds to \theta(g) = III...IZ g III..IZ
''' 
def knejaGlaser(g,element):
    k = []
    m = []
    
    for i in range(len(g)):
        elem = g[i]
        last = elem[len(elem)-1]
        if  (last == element) | (last == 0):
            m.append(elem)
        else:
            k.append(elem)
        
    return k,m




'''
    From an Abelian k and list of h, it produces a string of pauli terms to represent the first K element.
'''
def createK(abeliank, hlist):
    
    if len(hlist) == 0:
        return abeliank
    else:
        newhlist = hlist.copy()

        h = newhlist[0]
        newhlist.pop(0)
        
        oneless = createK(abeliank,newhlist)
        
        return  oneless + h + oneless 
    

'''
    Generates h with the order of the elements in m
'''
def getsubalgebra(m):
    h = []
    for i in range(len(m)):
        flag = 0
        for j in range(len(h)):
            comm = commutatePauliString(1, h[j], 1, m[i])
            if comm[0] != 0:
                flag = 1
                break
        if flag == 0:
            h.append(m[i]) 
            
    return h



'''
    Generates h starting with the elements in elemlist, with the order of the elements in m 
'''
def getsubalgebraelem(m, elemlist):
    h = elemlist.copy()
    for i in range(len(m)):
        flag = 0
        for j in range(len(h)):
            if m[i] == h[j]:
                flag = 1
                break
                
            comm = commutatePauliString(1, h[j], 1, m[i])
            if comm[0] != 0:
                flag = 1
                break
        if flag == 0:
            h.append(m[i]) 
            
    return h
   
    
'''
    Prints Pauli term lists with letters.
'''
def printlist(tuples):
    
    res = ''
    
    chars = 0
    for p in tuples:
        for i in range(len(p)):
            if p[i] == 0:
                res = res + 'I'
            elif p[i] == 1:
                res = res + 'X'
            elif p[i] == 2:
                res = res + 'Y'
            elif p[i] == 3:
                res = res + 'Z'
            chars = chars + 1
        
        res = res + ', '
        
        if chars > 140:
            res = res + '\n'
            chars = 0
        
    
    print(res+'\n\n\n')


'''
    Rounds coefficients that are smaller than accur to zero.
'''
def cleancoefs(coefs, accur):
    
    for i in range(len(coefs)):
        if abs(coefs[i])<accur:
            coefs[i] = 0
    
    
'''
    Prints the linear combination of pauli terms with coefs as coefficients and tuples as its terms.
'''
def printterms(coefs,tuples):
    
    res = ''
    
    chars = 0
    
    for j in range(len(coefs)):
        if coefs[j]!= 0:
            
            res = res + str(coefs[j]) + ' '
            
            p = tuples[j]

            for i in range(len(p)):
                if p[i] == 0:
                    res = res + 'I'
                elif p[i] == 1:
                    res = res + 'X'
                elif p[i] == 2:
                    res = res + 'Y'
                elif p[i] == 3:
                    res = res + 'Z'
                chars = chars + 1

            res = res + ' + '
            if chars > 50:
                res = res + '\n'
                chars = 0
    
    print(res+'\n\n\n')
    
'''
    Transforms numbers in tuples to Pauli terms
'''   
def paulilabel(p):
    res = ''
    for i in range(len(p)):
        if p[i] == 0:
            res = res + 'I'
        elif p[i] == 1:
            res = res + 'X'
        elif p[i] == 2:
            res = res + 'Y'
        elif p[i] == 3:
            res = res + 'Z'
    return res



'''
    Theta is just one number. k is just one Pauli term.
    Returns exp(i*theta*k)*(coefs*tuples)*exp(-i*theta*k) which is Adjoint action on the linear combination by thata*k.
'''
def adj_action(theta,k,coefs,tuples):
    
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


callcount = 0 



'''
   This returns Tr(exp(thetas1*k1)*v*exp(-thetas2*k2)*H)
   (To make it clear, for Earp and Pachos function, we have k1=k2=k, thetas1=thetas2=thetas)
'''
def funkygeneral(thetas1, thetas2, k_ints ,k, h, Hcoefs, Htuples, accur, index):  #index ==== how many elements will go for v, Htuples is an int list
    
    
    global callcount
    callcount = callcount + 1
    
    pi = math.pi
    hcoefs = [1]
    
    #generate v
    for i in range(len(h)-1):
        term = pi*hcoefs[i]
        while term > 1:
            term = term-1
        hcoefs.append(term)
      

    resH = [Hcoefs,Htuples] 
    resV = [hcoefs,range(len(k_tuples),len(k_tuples)+len(h_tuples))]
    #resV = [hcoefs,h]
    
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



'''
    returns Tr(exp(thetas*k)*v*exp(-thetas*k)*H)
'''
def funky(thetas, k_ints, k, h, Hcoefs, Htuples, accur):
    
    #k_ints = range(len(k))
    
    val = funkygeneral(thetas, thetas, k_ints, k, h, Hcoefs, Htuples, accur,-1)
    
    result = val.real
    global callcount
    #print(str(callcount) + ' ' + str(result))
    callcount = callcount + 1
    return result

'''
    returns gradient of funky. Order of derivatives is the order of the parameters thetas.
'''
def gradfunky(thetas, k_ints, k, h, Hcoefs, Htuples, accur):
    
    #k_ints = range(len(k))
    
    res = np.zeros(len(thetas))
    
    for i in range(len(thetas)):
        thetascopy = thetas.copy()
        thetascopy[i] = thetascopy[i]+math.pi/2
        
        diff = funkygeneral(thetascopy,thetas, k_ints, k, h, Hcoefs, Htuples, accur,i) + funkygeneral(thetas,thetascopy, k_ints, k, h, Hcoefs, Htuples, accur,i)
        
        res[i] = diff.real
        
    return res


'''
    returns Tr(exp(thetas*k)*v*exp(-thetas*k)*H)
'''
def funky2(thetas, k_ints, k, h, Hcoefs, Htuples, accur):
    
    
    thetas2 = np.zeros(2*len(thetas))
    for i in range(len(thetas)):
        thetas2[2*i] = thetas[i]
        thetas2[2*i+1] = -thetas[i]
    
    
    
    val = funkygeneral(thetas2, thetas2, k_ints, k, h, Hcoefs, Htuples, accur,-1)
    
    result = val.real
    global callcount
    #print(str(callcount) + ' ' + str(result))
    callcount = callcount + 1
    return result

'''
    returns gradient of funky. Order of derivatives is the order of the parameters thetas.
'''
def gradfunky2(thetas, k_ints, k, h, Hcoefs, Htuples, accur):
    
    
    thetas2 = np.zeros(2*len(thetas))
    for i in range(len(thetas)):
        thetas2[2*i] = thetas[i]
        thetas2[2*i+1] = -thetas[i]
    
    res = np.zeros(len(thetas))
        
    for i in range(len(thetas)):
        thetascopye = thetas2.copy()
        thetascopyo = thetas2.copy()
        
        thetascopye[2*i] = thetascopye[2*i]+math.pi/2
        thetascopyo[2*i+1] = thetascopyo[2*i+1]+math.pi/2
        
        diff = funkygeneral(thetascopye,thetas2, k_ints, k, h, Hcoefs, Htuples, accur,2*i) + funkygeneral(thetas2,thetascopye, k_ints, k, h, Hcoefs, Htuples, accur,2*i) - funkygeneral(thetascopyo,thetas2, k_ints, k, h, Hcoefs, Htuples, accur,2*i+1) - funkygeneral(thetas2,thetascopyo, k_ints, k, h, Hcoefs, Htuples, accur,2*i+1)
        
        res[i] = diff.real
      
    return res



'''
    Returns h = exp(-thetas*k)*H*exp(thetas*k)
'''
def gethVecFromk(thetas, k, Hcoefs, Htuples):
    

    I = (0,)*len(k[0])
    
    res = [Hcoefs, Htuples]
    
    #calculate exp(+...)*H*exp(-...)
    for i in range(len(thetas)):
        res = adj_action(-thetas[i],k[i],res[0],res[1])
        
    hcoefs = res[0]
    htuples = res[1]

    return hcoefs, htuples




'''
    Gets the norm square of the part in hcoefs*htuples that is orthogonal to Cartan subalgebra h.
'''
def errorhVec(hcoefs,htuples,h):
    
    result = 0
    
    for i in range(len(hcoefs)):
        
        term = htuples[i]
        flag = 0
        for j in range(len(h)):
            if term == h[j]:
                flag = 1
                break
        if flag == 0:
            result = result + abs(hcoefs[i])**2
            
    return result
    
    

'''
    Optimizes the Earp and Pachos function, which is given as funky here.
'''
def optimize(initialGuess, accuracy, steps, k_ints, k,h,Hcoefs,Htuples, accur, optimizerType, sym):
    #start_time = time.time()
    #Not used anymore
    #GlobalMin = scipy.optimize.shgo(function,[(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1),(-1,1)],args = (Hamiltonian,) , n = 200, iters = 4)
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    #Passes the Hamiltonian through the args, calle the "function" to minimize using the initial Guess we pass it initially

    if optimizerType == 'Powell':
        optimiumReturn = scipy.optimize.minimize(funky,initialGuess,args = (k_ints,k,h,Hcoefs,Htuples, accur) ,method='Powell',options={'disp':True, 'ftol':accuracy, 'maxiter':steps})
        #optimiumReturn = scipy.optimize.minimize(function,initialGuess,args = (Hamiltonian,) ,method='Powell',options={'disp':True, 'ftol':accuracy, 'maxiter':steps})
        # "xtol": accuracy})#, 'gtol':1e-20 ,'maxiter':500})
    elif optimizerType == 'BFGS':
        
        if sym == 'yes sym':
            optimiumReturn = scipy.optimize.minimize(funky2,initialGuess,args = (k_ints,k,h,Hcoefs,Htuples, accur) ,method='BFGS', jac = gradfunky2,options={'disp':True, 'gtol':accuracy, 'maxiter':steps})
            return optimiumReturn.x
        else:
            optimiumReturn = scipy.optimize.minimize(funky,initialGuess,args = (k_ints,k,h,Hcoefs,Htuples, accur) ,method='BFGS', jac = gradfunky,options={'disp':True, 'gtol':accuracy, 'maxiter':steps})
            return optimiumReturn.x
    else:
        optimiumReturn = scipy.optimize.minimize(function,initialGuess,args = (k_ints,k,h,Hcoefs,Htuples, accur), options={'disp':True, 'ftol':accuracy, 'maxiter':steps})# "xtol": accuracy})#, 'gtol'
    #Strips the result from the returned object of the minimizer

    x = optimiumReturn.x

    
    return(x)
        
    

'''
    HAMILTONIAN GENERATORS
'''

def hubbard(N):
    
    H = []
    for i in range(N-1):
        term = (0,)*i + (1,1) + (0,)*(2*N-i-2)         
        H.append(term)
        
        term = (0,)*i + (2,2) + (0,)*(2*N-i-2) 
        H.append(term)
        
        term = (0,)*(N+i) + (1,1) + (0,)*(N-i-2)         
        H.append(term)
        
        term = (0,)*(N+i) + (2,2) + (0,)*(N-i-2) 
        H.append(term)

    for i in range(N):
        term = (0,)*i + (3,) + (0,)*(N-1) + (3,) + (0,)*(N-i-1)  
        H.append(term)
        
    return H


def xymodel(N,bc):
    
    H = []
    
    for i in range(N-1):
        term = (0,)*i + (1,1) + (0,)*(N-i-2)         
        H.append(term)
        
        term = (0,)*i + (2,2) + (0,)*(N-i-2) 
        H.append(term)
        
    if bc == 'closed':
        term = (1,) + (0,)*(N-2) + (1,)
        H.append(term)
        
        term = (2,) + (0,)*(N-2) + (2,)
        H.append(term)
        
    return H

def xymodel_2nd(N,bc):
    
    H = []
    
    for i in range(N-1):
        term = (0,)*i + (1,1) + (0,)*(N-i-2)         
        H.append(term)
        
        term = (0,)*i + (2,2) + (0,)*(N-i-2) 
        H.append(term)
        
    for i in range(N-2):
        term = (0,)*i + (1,0,1) + (0,)*(N-i-3)         
        H.append(term)
        
        term = (0,)*i + (2,0,2) + (0,)*(N-i-3) 
        H.append(term)
        
    if bc == 'closed':
        term = (1,) + (0,)*(N-2) + (1,)
        H.append(term)
        
        term = (2,) + (0,)*(N-2) + (2,)
        H.append(term)
        
    return H



def kitaev(N,bc):
    
    H = []
    
    for i in range(N-1):
        if i%2==0:
            term = (0,)*i + (1,1) + (0,)*(N-i-2)         
            H.append(term)
        else:
            term = (0,)*i + (2,2) + (0,)*(N-i-2) 
            H.append(term)
        
    if (bc == 'closed') & (N%2==0):
        term = (2,) + (0,)*(N-2) + (2,)
        H.append(term)
        
    return H


def kitaevHoneyComb(length, count):
    
    
    N = length*count
    
    H = []
    
    for j in range(count):
        for i in range(length-1):
            if i%2==j%2:
                term = (0,)*(i+j*length) + (1,1) + (0,)*(N-(i+j*length)-2)         
                H.append(term)
            else:
                term = (0,)*(i+j*length) + (2,2) + (0,)*(N-(i+j*length)-2) 
                H.append(term)
        
        if j != count-1:
            for i in range(length):
                #print('hey')
                if i%2 == 1-j%2:
                    term = (0,)*(i+j*length) + (3,) + (0,)*(length-1)+(3,)+(0,)*(N-(i+(j+1)*length)-1)
                    H.append(term)
            
            
        
    return H

def heisenberg(N,bc):
    
    H = []
    
    for i in range(N-1):
        term = (0,)*i + (1,1) + (0,)*(N-i-2)         
        H.append(term)
        
        term = (0,)*i + (2,2) + (0,)*(N-i-2) 
        H.append(term)
        
        term = (0,)*i + (3,3) + (0,)*(N-i-2) 
        H.append(term)
        
    if bc == 'closed':
        term = (1,) + (0,)*(N-2) + (1,)
        H.append(term)
        
        term = (2,) + (0,)*(N-2) + (2,)
        H.append(term)
        
        term = (3,) + (0,)*(N-2) + (3,)
        H.append(term)
        
    
    return H

def tfim(N,bc):
    
    H = []
    
    for i in range(N):
        term = (0,)*i + (1,) + (0,)*(N-i-1) 
        H.append(term)
    
    for i in range(N-1):
        term = (0,)*i + (3,3) + (0,)*(N-i-2) 
        H.append(term)
        
    if bc == 'closed':
        term = (3,) + (0,)*(N-2) + (3,)
        H.append(term)
        
    return H  

def tfxy(N,bc):
    
    H = []
    
    for i in range(N):
        term = (0,)*i + (3,) + (0,)*(N-i-1) 
        H.append(term)
    
    for i in range(N-1):
        term = (0,)*i + (1,1) + (0,)*(N-i-2) 
        H.append(term)
        term = (0,)*i + (2,2) + (0,)*(N-i-2) 
        H.append(term)
        
    if bc == 'closed':
        term = (1,) + (0,)*(N-2) + (1,)
        H.append(term)
        term = (2,) + (0,)*(N-2) + (2,)
        H.append(term)
        
    return H  
    
  
    
    
'''
    MAIN METHODS
'''        

    
def getpaulirep(M,N):
    
    if N == 2:
        L = [0,0,0,0]
        L[0] = 0.5*(M[0][0]+M[1][1])     #I
        L[1] = 0.5*(M[0][1]+M[1][0])     #X
        L[2] = 0.5*1j*(M[0][1]-M[1][0])  #Y
        L[3] = 0.5*(M[0][0]-M[1][1])     #Z
        
        terms = [(0,),(1,),(2,),(3,)]
        
        index = 0
        return L,terms
    else:
        M11 = np.zeros((int(N/2),int(N/2)),dtype = complex)
        M12 = np.zeros((int(N/2),int(N/2)),dtype = complex)
        M21 = np.zeros((int(N/2),int(N/2)),dtype = complex)
        M22 = np.zeros((int(N/2),int(N/2)),dtype = complex)
        
        s = int(N/2)
        for i in range(int(N/2)):
            for j in range(int(N/2)):
                M11[i][j] = M[i][j]
                M12[i][j] = M[i][j+s]
                M21[i][j] = M[i+s][j]
                M22[i][j] = M[i+s][j+s]
                
        LI = getpaulirep(0.5*(M11+M22),int(N/2))
        LX = getpaulirep(0.5*(M12+M21),int(N/2))
        LY = getpaulirep(0.5*1j*(M12-M21),int(N/2))
        LZ = getpaulirep(0.5*(M11-M22),int(N/2))
    
        L = LI[0] + LX[0] + LY[0] + LZ[0]
        
        for i in range(len(LI[1])):
            LI[1][i] = (0,) + LI[1][i]  
            LX[1][i] = (1,) + LX[1][i]
            LY[1][i] = (2,) + LY[1][i]
            LZ[1][i] = (3,) + LZ[1][i]
            
        terms = LI[1] + LX[1] + LY[1] + LZ[1]
        return L,terms
        
    
    

    
def tfxyresults(N, accur,partial, devcount, maxdev, rep, sym):
    
    accuracy = accur
    #accuracy = 0.00001
    #accur = 0.00001
    steps = 4500
    
    Htuples = tfxy(N,'open')
    
    print('H:\n')
    printlist(Htuples)
    
    g = makeGroup(Htuples)
    
    [k,m] = elemcount(g,2)
    
    h = getsubalgebra(m)
    
    set_tuples(k,m,h)
    
    k = []
    
    if rep == 'cascade':
        for i in range(N):
            for j in range(N-i-1):
                elem = (0,)*i + (2,) + (3,)*j + (1,) + (0,)*(N-j-i-2)
                k.append(elem)
                
                
                elem = (0,)*i + (1,) + (3,)*j + (2,) + (0,)*(N-j-i-2)
                k.append(elem)
                '''
        for i in range(N):
            for j in range(N-i-1):
                elem = (0,)*i + (1,) + (3,)*j + (2,) + (0,)*(N-j-i-2)
                k.append(elem)
'''
        
    elif rep == 'pile':
        for i in range(N-1):
            for j in range(N-i-1):
                elem = (0,)*j+(2,1)+(0,)*(N-j-2)
                k.append(elem)
                
                elem = (0,)*j+(1,2)+(0,)*(N-j-2)
                k.append(elem)
                

            
    
    k_ints = [] #List of indices for the k tuples in g?
    for i in range(len(k)):
        for j in range(len(g_tuples)):
            if k[i] == g_tuples[j]:
                k_ints.append(j)
                break
            
    print(k_ints)        
            
    
    
    Htuplesint = [] #List of indices for the H tuples in g?
    for i in range(len(Htuples)):
        for j in range(len(g_tuples)):
            if g_tuples[j]==Htuples[i]:
                Htuplesint.append(j)
                break
    
    
    print('k:\n')
    printlist(k)
    print('m:\n')
    printlist(m)
    print('h:\n')
    printlist(h)
    
    
    initialGuess = np.zeros(len(k))
    if sym == 'yes sym':
        initialGuess = np.zeros(int(len(k)/2))
    
    
    Hcoefs0 = np.ones(len(Htuples)) #No Idea
    for j in range(N):
        Hcoefs0[j] = 0
    
    perturb = np.random.normal(0,1,len(Htuples)) #Leave out for now
    for j in range(N,len(Htuples)):
        #print(j)
        perturb[j] = 0
        
    print(perturb)
    print(Hcoefs0)
    

    
    
    Bcount = devcount #?????
    
    
    errors = []
    times = []
    hcoefslist = []
    
    angles = []
    B = [] #?????
    
    
    for i in range(Bcount): #????
        
        std = maxdev*i/(Bcount-1)
        print(std)
    
    
    
    for i in range(Bcount):
        
        std = maxdev*i/(Bcount-1)
        Hcoefs = Hcoefs0 + std * perturb #Generates the hamiltonian
        B.append(std)
        
        total_time = time.time()
        thetas = optimize(initialGuess, accuracy, steps, k_ints,k,h,Hcoefs,Htuplesint,0,'BFGS',sym)
        #thetas = optimize(initialGuess, accuracy, steps, k,h,Hcoefs,Htuplesint,0,'Powell')
        timepassed = time.time() - total_time
        print('--- ' + str(timepassed) +  ' seconds ---')
        
        times.append(timepassed)
        
        initialGuess = thetas
        
        thetas2 = np.copy(thetas)
        
        if sym == 'yes sym':
            thetas2 = np.zeros(2*len(thetas))
            for w in range(len(thetas)):
                thetas2[2*w] = thetas[w]
                thetas2[2*w+1] = -thetas[w]
                
        
        angles.append(thetas2)
    
        [hcoefs,htuples] = gethVecFromk(thetas2, k, Hcoefs, Htuples)
    
        hcfs = []
        
        for w in h:
            hflag = 0
            for u in range(len(htuples)):
                if w == htuples[u]:
                    hcfs.append(hcoefs[u])
                    hflag = 1
                    break
            if hflag == 0:
                hcfs.append(0)
        
        hcoefslist.append(hcfs)
        '''
        print(hcoefs)
        printlist(htuples)
        printlist(h)
        '''
    
    
        error1 = errorhVec(hcoefs,htuples,h)
        cleancoefs(hcoefs,accur)
             
    
        error2 = errorhVec(hcoefs,htuples,h)
        print('Error1: ' + str(error1) + '\tError2: ' + str(error2))  
              
        errors.append(error1)
        
        if partial == 'partial':
            #filename = 'current_running/'+'tfxy_withminus_step'+str(i)+'_outof_'+str(Bcount)
            filename = 'current_running/'+'tfxy_step'+str(i)+'_outof_'+str(Bcount)
            print(filename)
            np.savez(filename,k = k, m = m, h = h, inter = B, angles = angles, errors = errors, times = times)
            npzfile = np.load(filename+'.npz')    
            print(npzfile['errors'])
            print(npzfile['times'])

            
    print(B)
    
    
    finangles = []
    for i in range(len(angles[0])):
        finangles.append([])
    
    for i in range(len(angles)):
        for j in range(len(angles[0])):
            finangles[j].append(angles[i][j])
    
    for i in range(len(finangles)):
        plt.plot(B,finangles[i], label = paulilabel(k[i]))
              
    #filename = 'current_running/'+str(N)+'tfxy_withminus'
    filename = str(N)+'tfxy'
    np.savez(filename,k = k, m = m, h = h, inter = B, angles = finangles, hcoefs = hcoefslist, errors = errors, times = times, Htuples = Htuples, Hcoefs0 = Hcoefs0, perturb = perturb)
    
    plt.legend()
    plt.show()
    plt.savefig(filename, dpi=None, facecolor='w', edgecolor='w', orientation='portrait', papertype=None, format=None, transparent=False, bbox_inches=None, pad_inches=0.1, frameon=None, metadata=None)
    

    
    
tfxyresults(4, 0.001,'no', 5, 4, 'pile', 'yes')
