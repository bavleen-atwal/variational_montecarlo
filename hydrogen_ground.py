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

def local_energy_3d(psi, v, rho, h, r_eps = 1e-12, psi_eps = 1e-300):
    """Finding local energy of wavefunction in 3D using laplacian and central difference scheme"""
    v = np.asarray(v, dtype=float)
    r = np.linalg.norm(v)
    r = max(r, r_eps)
    psi0 = psi(v, rho)
    psi0 = max(abs(psi0), psi_eps)
    lap = laplacian_central(psi, v, rho, h)
    kinetic = -0.5* (lap / psi0)
    potential = -1.0 / r
    return kinetic + potential

def estimate_energy_3d(psi, samples, rho, h, show=True):
    """Finding estimated energy from sampled metropolis x values in 3D"""

    r_samples = np.asarray(samples, dtype=float)

    E_samples = np.array([local_energy_3d(psi, v, rho, h) for v in r_samples], dtype=float)

    E_mean = np.mean(E_samples)
    E_std = float(np.std(E_samples, ddof=1))
    E_sem = E_std / np.sqrt(len(E_samples))
    if show:
        print(f'Estimated energy: {E_mean:.6f} ± {E_sem:.3f}')
        plt.hist(E_samples, bins=90, density=True, alpha=0.7, color='maroon')
        plt.xlabel(r"Local energy $E_L$")
        plt.ylabel("Density")
        plt.title("Local energy distribution")
        plt.figtext(0.55, 0.8, f'Estimated mean energy: {E_mean:.3f}', fontsize=10)
        plt.show()
    return E_samples, E_mean, E_sem

def initial_scan(psi, rhos, x0, nsteps, stepsize, nburn, h_lap, seed, doplot = False):
    """First scan over variational parameter rho to find initial energy curve. Checks if current pipeline is converging"""

    rhos_out = np.asarray(rhos, dtype=float)
    E_means = np.zeros_like(rhos_out)
    E_sems = np.zeros_like(rhos_out)
    acc_rates = np.zeros_like(rhos_out)
    for i, rho in enumerate(rhos_out):
        accepted_x, full_x, acc_rate, r_trace = metropolis_3d(
            psi=psi_hydrogen,
            x0=x0,
            rho=rho,
            nsteps=nsteps,
            stepsize=stepsize,
            nburn=nburn,
            seed=seed
        )
        E_samples, E_mean, E_sem = estimate_energy_3d(
            psi=psi,
            samples=accepted_x,
            rho=rho,
            h=h_lap,
            show=False
        )
        E_means[i] = E_mean
        E_sems[i] = E_sem
        acc_rates[i] = acc_rate
        print(f"rho={rho:.3f}  E={E_mean:.6f} ± {E_sem:.6f}  acc={acc_rate:.3f}")

    if doplot:
        plt.errorbar(rhos_out, E_means, yerr=E_sems, fmt='o', capsize=3, color='k')
        plt.xlabel(r"Variational parameter $\rho$")
        plt.ylabel(r"Estimated energy $E(\rho)$")
        plt.title(r"VMC energy curve $E(\rho)$")
        plt.show() 
    
    pack = [rhos_out, E_means, E_sems, acc_rates]
    return pack

# Diagnostic functions for optimising and verifying

def laplacian_diagnostic(a = 0.7, seed=0, npoints=200, show=True):

    # Defining known gaussian function
    def phi_gaussian(v, a):
        r2 = np.dot(v, v)
        return np.exp(-a * r2)
    
    # Defining known analytic laplacian
    def laplacian_gaussian_analytic(v, a):
        r2 = np.dot(v, v)
        return (-6.0 * a + 4.0 * a * a * r2) * np.exp(-a * r2)
    
    rng = np.random.default_rng(seed)

    # Random test points
    points = rng.normal(0.0, 1.0, size=(npoints, 3))

    hs = np.array([1e-5,1e-4, 2e-4, 3e-4, 1e-3, 2e-3, 3e-3, 1e-2, 2e-2, 3e-2, 1e-1, 7e-2, 5e-2, 3e-2, 2e-2, 1.5e-2, 1e-2])
    errors = []

    #Computing errors for different step sizes
    for h in hs:
        err_list = []
        for v in points:
            fd = laplacian_central(phi_gaussian, v, a, h)
            an = laplacian_gaussian_analytic(v, a)
            err_list.append(abs(fd - an))
        errors.append(np.mean(err_list))

    errors = np.array(errors)
    p, c = np.polyfit(np.log(hs), np.log(errors), 1)
    print(f"Estimated order p ≈ {p:.3f} (expect ~2 for central 2nd derivative)")
    if show:
        plt.loglog(hs, errors, marker='o', color='k', linestyle = '--', label = f'Slope of {p:.3f}')
        plt.xlabel("Finite-difference step h")
        plt.ylabel("Mean absolute error")
        plt.title("3D Laplacian verification for truncation error")
        plt.legend()
        plt.show()

    # Generating random points again for set step size
    step = 1e-2
    rng = np.random.default_rng(1)
    points = rng.normal(0.0, 1.0, size=(5, 3))

    print("v\t\tFD Laplacian\tExact Laplacian\tAbsolute error")
    print("-"*70)

    # Comparing finite-difference and analytic laplacians directly
    for v in points:
        fd = laplacian_central(phi_gaussian, v, a, step)
        exact = laplacian_gaussian_analytic(v, a)
        err = abs(fd - exact)
        print(f"{v}\t{fd: .6e}\t{exact: .6e}\t{err: .2e}")

    return hs, errors

def burn_in_diagnostic(r_trace, nburn, stepsize=None):
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

def sampling_diagnostic(r_trace, nburn, rho, bins=50):
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
h = 1e-4
rhos = np.array([0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5])

test_x, testfull_x, test_rate, test_trace = metropolis_3d(
    psi=psi_hydrogen,
    x0=x0,
    rho=rho,
    nsteps=nsteps,
    stepsize=stepsize,
    nburn=nburn,
    seed = 1234
)

test_E, test_mean, test_sem = estimate_energy_3d(psi_hydrogen, test_x, rho, h=h, show=False)

pack = initial_scan(psi_hydrogen, rhos, x0, nsteps, stepsize, nburn, h_lap=h, seed=1234, doplot=True)


print("Acceptance rate:", test_rate)
print("full_x shape:", testfull_x.shape)

# Plotting diagnostics to select burn-in and check correct sampling
#burn_in_diagnostic(r_trace, nburn, stepsize)
#sampling_diagnostic(r_trace, nburn, rho)
#laplacian_diagnostic()