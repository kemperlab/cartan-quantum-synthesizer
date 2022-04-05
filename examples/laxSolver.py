from CQS.methods import Hamiltonian, Cartan, FindParameters
import numpy as np

ham = Hamiltonian(2, [(1, 'tfim', False)])
cartan = Cartan(ham, involution='countY')
FindParameters(cartan, optimizerMethod='Lax')



"""from scipy.integrate import ode

x0 = np.array([1,1,1,1])
#y0 = np.array([1,1])

def func(t, z):
    return np.array([1,1,1,1])

integrator = ode(func).set_integrator('dopri5')
integrator.set_initial_value(x0, 0)
dt = 0.1
integrator.integrate(integrator.t + dt)
"""