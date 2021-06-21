# -*- coding: utf-8 -*-
__docformat__ = 'google'
'''import sys
sys.path.append('c:\\Users\\Thoma\\OneDrive\\Documents\\2021_ORNL\\CartanCodeGit\\cartan-quantum-synthesizer')
print(sys.path)'''

from package.methods.Hamiltonian import Hamiltonian
from package.methods.Cartan import Cartan

xy = Hamiltonian(4,[(1,'xy')])
xyC = Cartan(xy)
print(xyC.k)
print(xyC.m)
print(xyC.g)
print(xyC.h)
xyC.subAlgebra(seedList=[(0,1,1,0),(0,2,2,0)])
print(xyC.h)

xyC.decompose('knejaGlaser')
print(xyC.k)


print('\n Six Site XY \n')
xy = Hamiltonian(6,[(1,'xy')])
print('Hamiltonian')
print(xy.HTuples)
xyC = Cartan(xy)
print('K')
print(xyC.k)
print('M')
print(xyC.m)
print("G")
print(xyC.g)
print('h')
print(xyC.h)