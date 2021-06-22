# -*- coding: utf-8 -*-
__docformat__ = 'google'
'''import sys
sys.path.append('c:\\Users\\Thoma\\OneDrive\\Documents\\2021_ORNL\\CartanCodeGit\\cartan-quantum-synthesizer')
print(sys.path)'''
from methods import Hamiltonian

print('generic XY')
xymodel = Hamiltonian(4, name=[(1, 'xy')])
print(xymodel.getHamiltonian())
print(xymodel.getHamiltonian(type='text'))
xymodel.getHamiltonian(type='printText')
xymodel.getHamiltonian(type='printTuples')

print('custom coefficients')
xymodel = Hamiltonian(4, name=[([1,2,3,4,5,6], 'xy')])
print(xymodel.getHamiltonian())


print('+1 mismatch')
try:
    xymodel = Hamiltonian(4, name=[([1,2,3,4,5,6,7], 'xy')])
except:
    print("passed + 1 mismatch")

print('-1 mismatch')
try:
    xymodel = Hamiltonian(4, name=[([1,2,3,4,5], 'xy')])
except:
    print("passed - 1 mismatch")

print('Combine Models')
xymodel = Hamiltonian(4, name=[([1,2,3,4,5,6,7,8], 'xy', True),(1,'transverse_z')])
print(xymodel.getHamiltonian())