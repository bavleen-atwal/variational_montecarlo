
import numpy as np
import matplotlib.pyplot as plt

def central_diff(f, x, h):
    """Compute the central difference approximation of the derivative of f at x."""
    return (f(x + h) - 2*f(x) + f(x - h)) / (h**2)

def local_energy(psi, x, h):

    kinetic = -0.5 * central_diff(psi, x, h) / psi(x)
    potential = 0.5 * x**2
    return kinetic + potential

# Finding central difference error for ground state

def psi_0(x):
    return np.exp(-0.5* x**2)

x0 = 0.0
x1 = 1.0

psi0_expect = -1.0
psi1_expect = 0.0

hs = np.logspace(-1, -6, num=12)

steperror: list[tuple[float, float]] = []

for h in hs:
    foundpsi = central_diff(psi_0, x0, h)
    err = np.abs(foundpsi - psi0_expect)
    steperror.append((h, err))

plt.loglog(*zip(*steperror), 'ko--', linewidth=1, markersize=4)
plt.xlabel('Step size h')
plt.ylabel('Error in derivative at x=0')
plt.title('Central Difference Error for Second Derivative Ground state')
plt.show()
