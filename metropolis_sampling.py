import numpy as np
import matplotlib.pyplot as plt

def psi_0(x):
    """Ground state wavefunction"""
    return np.exp(-0.5* x**2)

def logP(psi, x):
    """Taking the log of the unnormalised target probability distribution"""
    if psi(x) == 0:
        return -np.inf
    return 2 * np.log(abs(psi(x)))

def psi_second(psi, x, h=1e-4):
    """Second derivative of psi using central difference"""
    return (psi(x + h) - 2*psi(x) + psi(x - h)) / (h**2)

def local_energy(psi, x):
    """Finding the local energy of wavefunction at x"""
    kinetic = -0.5 * psi_second(psi, x) / psi(x)
    potential = 0.5 * x**2
    return kinetic + potential

def metropolis_sampling(psi, x0, nsteps, nburn, stepsize, seed = None):
    """Metropolis sampling of PDF to generate set of x values"""

    rng = np.random.default_rng(seed)
    accepted_x = []
    full_x = []
    accepted = 0
    x = x0
    logP_x = logP(psi, x)
    
    for i in range(nsteps):
        x_trial = x + rng.normal(0.0, stepsize)
        logP_trial = logP(psi, x_trial)

        log_r = logP_trial - logP_x
        log_u = np.log(rng.random())

        if log_r >= 0.0 or log_u < log_r:
            x = x_trial
            logP_x = logP_trial
            accepted += 1
        full_x.append(x)

        if i>= nburn:
            accepted_x.append(x)

        else:
            pass
    acceptance_rate = accepted / nsteps
    return accepted_x, acceptance_rate, full_x

def estimate_energy(psi, samples, full_x):
    """Estimating the local energy from sampled x values"""

    E_samples = np.array([local_energy(psi, x) for x in samples], dtype = float)
    E_full = np.array([local_energy(psi, x) for x in full_x])
    running_mean = np.cumsum(E_full) / np.arange(1, len(E_full)+1)
    E_mean = np.mean(E_samples)
    E_var = np.var(E_samples)
    return E_samples, E_mean, E_var, running_mean

def autocorrelation(x, max_lag=100):
    x = np.asarray(x)
    x_mean = np.mean(x) #Centring to find fluctuations around the mean
    var = np.mean((x - x_mean)**2) #Computing variance
    ac = []
    for lag in range(max_lag):
        c = np.mean((x[:-lag or None] - x_mean) *   #Covariance between x separated by lag steps
                    (x[lag:] - x_mean))
        ac.append(c / var) #Normalising
    return np.array(ac)

def integrated_autocorrelation_time(ac): #Quantifying autocorrelation time
    return 0.5 + np.sum(ac[1:])

nburn = 5000
sigmasize = 1.0
# First run pass

x_samples, acc_rate, full_x = metropolis_sampling(psi_0, x0=10.0, nsteps=100000, nburn = nburn, stepsize = sigmasize, seed = 12345)

E_samples, E_mean, E_var, running_mean = estimate_energy(psi_0, x_samples, full_x)

print(f"Acceptance Rate: {acc_rate:.4f}")
print(f"Estimated Energy: {E_mean:.6f} ± {np.sqrt(E_var/len(E_samples)):.6f}")


plt.plot(x_samples, E_samples, 'k.')
plt.xlabel('Sampled x values')
plt.ylabel('Local Energy')
plt.title('Local Energy vs Sampled x values')
plt.show()

plt.plot(full_x, color = 'k')
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.title('Burn-in diagnostic plot, x0 = 10.0')
plt.show()

plt.plot(running_mean[:nburn])
plt.axhline(0.5, linestyle='--')
plt.xlabel("Iteration")
plt.ylabel("Running mean of E_L")
plt.title("Burn-in diagnostic: running mean")
plt.show()

ac = autocorrelation(full_x[nburn:], max_lag=50)
tau_int = integrated_autocorrelation_time(ac)
print("Integrated autocorrelation time:", tau_int)
plt.plot(ac, '.--', color = 'k')
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.title(f"Autocorrelation of x for stepsize of {sigmasize}")
plt.legend([f"τ_int = {tau_int:.2f}"])
plt.show()