# -*- coding: utf-8 -*-
__docformat__ = 'google'
'''import sys
sys.path.append('c:\\Users\\Thoma\\OneDrive\\Documents\\2021_ORNL\\CartanCodeGit\\cartan-quantum-synthesizer')
#print(sys.path)'''
from cqs.methods import Hamiltonian, Cartan, FindParameters

import numpy as np
from cqs.util.IO import tuplesToMatrix
from cqs.util.verification import Trotter, exactU, PauliExpUnitary, KHK

sites = 6
randCoNumpy = np.random.rand(16)
randCoList = []
for i in randCoNumpy:
    randCoList.append(i)
model = [(randCoList,'tfxy', False)]
xy = Hamiltonian(sites,model)
print(xy.getHamiltonian())

xyC = Cartan(xy)
xyC.decompose('countY')
xyP = FindParameters(xyC, optimizerMethod='BFGS')
xyP.printResult()

finalTime = 1
U_cartan = KHK(xyP.kCoefs, np.multiply(xyP.hCoefs,finalTime),xyC.k, xyC.h)

U_exact = exactU(xy.HCoefs, xy.HTuples, finalTime)

print(np.linalg.norm(U_exact - U_cartan))
