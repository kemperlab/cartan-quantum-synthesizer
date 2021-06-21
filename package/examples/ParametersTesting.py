import sys
sys.path.append('c:\\Users\\Thoma\\OneDrive\\Documents\\2021_ORNL\\CartanCodeGit\\cartan-quantum-synthesizer')
#print(sys.path)
from package.src.Hamiltonian import Hamiltonian
from package.src.Cartan import Cartan
from package.src.FindParameters import FindParameters
import numpy as np
from package.util.IO import tuplesToMatrix
from package.util.verification import Trotter, exactU, PauliExpUnitary, KHK

sites = 5
model = [(1,'tfxy', True)]

xy = Hamiltonian(sites,model)
xyC = Cartan(xy)
xyC.decompose('countY')
xyP = FindParameters(xyC, optimizerMethod='BFGS')
xyP.printResult()

finalTime = 1
U_cartan = KHK(xyP.kCoefs, np.multiply(xyP.hCoefs,finalTime),xyC.k, xyC.h)

U_exact = exactU(xy.HCoefs, xy.HTuples, finalTime)

print(np.linalg.norm(U_exact - U_cartan))