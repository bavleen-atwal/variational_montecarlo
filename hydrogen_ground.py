import numpy as np
import matplotlib.pyplot as plt


def psi_hydrogen(v, rho):
    """Trial wavefunction for hydrogen atom"""

    r = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    return np.exp(-rho*r)

def logP_3d(psi, v, rho):
    """Taking the log of the unnormalised target probability distribution in 3D"""
    if psi(v, rho) == 0:
        return -np.inf
    return 2 * np.log(abs(psi(v, rho)))

def metropolis_3d(psi, x0, rho, nsteps, stepsize,nburn, seed  = None):
    """
    3D Metropolis sampler using symmetric Gaussian proposals """
    rng = np.random.default_rng(seed)
    full_x = []
    accepted = 0
    accepted_x = []

    x = np.array(x0, dtype=float)
    logP_x = logP_3d(psi, x, rho)

    full_x.append(x.copy())
    r_trace = [np.linalg.norm(x)]

    for i in range(nsteps):
        x_trial = x + rng.normal(0.0, stepsize, size=3)

        logP_trial = logP_3d(psi, x_trial, rho)

        log_r = logP_trial - logP_x
        log_u = np.log(rng.random())

        if (log_r >= 0.0) or (log_u < log_r):
            x = x_trial
            logP_x = logP_trial
            accepted += 1

        full_x.append(x.copy())
        r_trace.append(np.linalg.norm(x))
        if i>= nburn:
            accepted_x.append(x.copy())

        else:
            pass

    acceptance_rate = accepted / nsteps
    return accepted_x, np.array(full_x), acceptance_rate, np.array(r_trace)  

def laplacian_central(psi, v, rho, h):
    """Laplacian of wavefunction using central difference"""
    v = np.asarray(v, dtype=float)
    psi0= psi(v, rho)
    lap = 0.0
    for i in range(3):
        e = np.zeros(3)
        e[i] = 1.0
        v_plus = v + h * e
        v_minus = v - h * e
        lap += (psi(v_plus, rho) - 2 * psi0 + psi(v_minus, rho)) / (h**2)
    
    return lap

def burn_in_diagnostic(r_trace, nburn):
    """Plotting running mean of r to diagnose burn-in period"""
    running_mean = np.cumsum(r_trace) / np.arange(1, len(r_trace) + 1)
    plt.plot(running_mean, lw=1.2, color='k')
    plt.xlabel(f"Metropolis step of {stepsize} sigma")
    plt.ylabel(r"Running mean of $r$")
    plt.axvline( x=nburn, color='r', linestyle='--', label=f'Iteration {nburn}, End of burn-in')
    plt.title("Running mean of $r$ (burn-in diagnostic)")
    plt.xlim(right=50000)
    plt.legend()
    plt.show()

def sampling_diagnostic(r_trace, rho, bins=50):
    r_samples = r_trace[nburn:]
    counts, bins, _ = plt.hist(
        r_samples,
        bins=50,
        density=True,
        alpha=0.7,
        label="Sampled $r$"
    )
    plt.xlabel(r"$r$")
    plt.ylabel("Probability density")
    r_vals = np.linspace(0, r_samples.max(), 400)
    theory = r_vals**2 * np.exp(-2 * rho * r_vals)
    theory /= np.trapz(theory, r_vals)
    plt.plot(r_vals, theory, 'r--', lw=2, label=r"$r^2 e^{-2\rho r}$")
    plt.legend()
    plt.title("Radial distribution check")
    plt.show()

# Testing

x0 = np.array([0.5, 0.0, 0.0])
rho = 1.0
nsteps = 100000
stepsize = 1.0
nburn = 6000

accepted_x, full_x, acc_rate, r_trace = metropolis_3d(
    psi=psi_hydrogen,
    x0=x0,
    rho=rho,
    nsteps=nsteps,
    stepsize=stepsize,
    nburn=nburn,
    seed = 1234
)

print("Acceptance rate:", acc_rate)
print("full_x shape:", full_x.shape)

# Plotting diagnostics to select burn-in and check correct sampling
burn_in_diagnostic(r_trace, nburn)
sampling_diagnostic(r_trace, rho)