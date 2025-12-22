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
            psi=psi,
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

    if doplot:
        print(f"rho={rho:.3f}  E={E_mean:.6f} ± {E_sem:.6f}  acc={acc_rate:.3f}")

        plt.errorbar(rhos_out, E_means, yerr=E_sems, fmt='o', capsize=3, color='k')
        plt.xlabel(r"Variational parameter $\rho$")
        plt.ylabel(r"Estimated energy $E(\rho)$")
        plt.title(r"VMC energy curve $E(\rho)$")
        plt.show() 
    
    pack = [rhos_out, E_means, E_sems, acc_rates]
    return pack

def vmc_E_of_rho(psi, rho, x0, nsteps, stepsize, nburn, h, seed=None):
    """ Runtime function for VMC """
    accepted_x, full_x, acc_rate, _ = metropolis_3d(
        psi=psi,
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
        h=h,
        show=False
    )
    return float(E_mean), float(E_sem), float(acc_rate)

def bracket_from_scan(pack):
    """ From initial scan data, finds 3 rho values bracketing minimum energy """
    rhos_out, E_means, E_sems, acc_rates = pack
    rhos_out = np.asarray(rhos_out, dtype=float)
    E_means = np.asarray(E_means, dtype=float)

    j = int(np.argmin(E_means))

    if j == 0 or j == len(rhos_out) - 1:
        raise ValueError(
            "Extend the scan range so the minimum is bracketed."
        )

    return float(rhos_out[j - 1]), float(rhos_out[j]), float(rhos_out[j + 1])

def parabolic_minimise_rho(
    psi,
    x0,
    nsteps,
    stepsize,
    nburn,
    h,
    rho1,
    rho2,
    rho3,
    seed=1234,
    max_iter=12,
    tol_rho=1e-3,
    min_step=1e-4,
    verbose=True
):
    """ Parabolic interpolation minimiser for 1D variational parameter rho."""
    cache = {}
    def cached_vmc(rho_val, seed_val):
        key = int(round(float(rho_val) / min_step))
        if key in cache:
            if verbose:
                print("Cache hit")
            return cache[key]
        En, s, a = vmc_E_of_rho(psi, rho_val, x0, nsteps, stepsize, nburn, h, seed=seed_val)
        cache[key] = (En, s, a)
        return En, s, a

    # Evaluating initial bracket
    E1, s1, a1 = cached_vmc(rho1, seed_val=seed)
    E2, s2, a2 = cached_vmc(rho2, seed_val=seed)
    E3, s3, a3 = cached_vmc(rho3, seed_val=seed)

    if not (rho1 < rho2 < rho3):
        raise ValueError("Require rho1 < rho2 < rho3.")

    history = []

    def parabola_vertex(x1, y1, x2, y2, x3, y3):
        """
        x-coordinate of the minimum of the parabola through three points.
        """
        # Denominator
        denom = (x2 - x1) * (y2 - y3) - (x2 - x3) * (y2 - y1)
        if not np.isfinite(denom) or abs(denom) < 1e-14:
            return None

        num = (x2 - x1)**2 * (y2 - y3) - (x2 - x3)**2 * (y2 - y1)
        x_min = x2 - 0.5 * (num / denom)

        if not np.isfinite(x_min):
            return None

        return float(x_min)

    if verbose and not (E2 <= E1 and E2 <= E3):
        print("Warning: initial bracket does not satisfy E(rho2) <= E(rho1), E(rho3). "
              "Noise may be large; consider increasing nsteps or scanning a bit wider.")

    rho_best = rho2
    E_best = E2
    it=0
    while (rho3-rho1) >= tol_rho and it < max_iter:
        it += 1
        rho_new = parabola_vertex(rho1, E1, rho2, E2, rho3, E3)

        # Fallback if parabola fit is degenerate
        if rho_new is None or not np.isfinite(rho_new):
            rho_new = 0.5 * (rho1 + rho3)

        if rho_new <= rho1 or rho_new >= rho3: # Clamping to window
            rho_new = 0.5 * (rho1 + rho3)

        if abs(rho_new - rho2) < min_step: # Force move step if new proposed sample is too close to old x
            direction = 1.0 if rho_new >= rho2 else -1.0
            rho_new = rho2 + direction * min_step

        E_new, s_new, a_new = cached_vmc(rho_new, seed_val=seed)

        # Recording iteration state
        history.append({
            "iter": it,
            "rho1": rho1, "E1": E1,
            "rho2": rho2, "E2": E2,
            "rho3": rho3, "E3": E3,
            "rho_new": rho_new, "E_new": E_new,
            "acc_new": a_new
        })

        if verbose:
            print(f"[{it:02d}] bracket: ({rho1:.6f}, {rho2:.6f}, {rho3:.6f}) "
                  f"E=({E1:.6f}, {E2:.6f}, {E3:.6f})  -> rho_new={rho_new:.6f}, E_new={E_new:.6f}")

        if E_new < E_best:
            rho_best, E_best = rho_new, E_new

        old_bracket = (rho1, rho2, rho3)

        pts = [(rho1, E1), (rho2, E2), (rho3, E3), (rho_new, E_new)] #Adding new point to bracket set
        worst = max(pts, key=lambda t: t[1])
        pts.remove(worst)
        pts.sort(key=lambda t: t[0])
        (rho1, E1), (rho2, E2), (rho3, E3) = pts

        if (rho1, rho2, rho3) == old_bracket:
            if (rho2 - rho1) > (rho3 - rho2):
                rho_try= 0.5 * (rho1+rho2)
                E_try, s_try, a_try = cached_vmc(rho_try, seed_val=seed)
                rho1, E1 = rho_try, E_try
            else:
                rho_try = 0.5 * (rho2+rho3)
                E_try, s_try, a_try = cached_vmc(rho_try, seed_val=seed)
                rho3, E3 = rho_try, E_try
            pts_try = [(rho1, E1), (rho2, E2), (rho3, E3)]
            pts_try.sort(key=lambda t: t[0])
            (rho1, E1), (rho2, E2), (rho3, E3) = pts_try
        
        if (rho3 - rho1) < tol_rho:
            candidates = [(rho1, E1), (rho2, E2), (rho3, E3)]
            rho_star, E_star = min(candidates, key=lambda t: t[1])
            if verbose:
                print(f"Converged: rho_best={rho_best:.6f}, E_best={E_best:.6f}")
                print(
                    f"Converged: bracket=({rho1:.6f}, {rho2:.6f}, {rho3:.6f})  "
                    f"-> rho*={rho_star:.6f}, E*={E_star:.6f}"
                )
            break

    return rho_star, E_star, history

############# Diagnostic functions for optimising and verifying ############

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

def plot_xy_density(accepted_x, bins=150, density=True, title=None):
    """ Projected number density in the x–y plane from 3D Metropolis samples. """
    samples = np.asarray(accepted_x, dtype=float)
    x = samples[:, 0]
    y = samples[:, 1]

    H, xedges, yedges, im = plt.hist2d(x, y, bins=bins, density=density)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.colorbar(im, label="Projected density" if density else "Counts per bin")

    if title is None:
        title = "Ground-state number density projected onto x–y plane"
    plt.title(title)
    plt.show()

# Testing

x0 = np.array([0.5, 0.0, 0.0])
rho = 1.0
nsteps = 100000
stepsize = 1.0
nburn = 6000
h = 1e-4
rhos = np.array([0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5])

pack = initial_scan(psi_hydrogen, rhos, x0, nsteps, stepsize, nburn, h_lap=h, seed=1234, doplot=False)
rho1, rho2, rho3 = bracket_from_scan(pack)

test_x, testfull_x, test_rate, test_trace = metropolis_3d(
    psi=psi_hydrogen,
    x0=x0,
    rho=rho,
    nsteps=nsteps,
    stepsize=stepsize,
    nburn=nburn,
    seed = 1234
)
plot_xy_density(test_x, bins=150, density=True)


rho_opt, E_opt, hist = parabolic_minimise_rho(
    psi=psi_hydrogen,
    x0=x0,
    nsteps=nsteps,
    stepsize=stepsize,
    nburn=nburn,
    h=h,
    rho1=rho1,
    rho2=rho2,
    rho3=rho3,
    seed=1234,
    max_iter=35,
    tol_rho=1e-3,
    verbose=True
)

print("\nOptimal rho:", rho_opt)
print("Optimal energy:", E_opt)

# Plotting diagnostics to select burn-in and check correct sampling
#burn_in_diagnostic(r_trace, nburn, stepsize)
#sampling_diagnostic(r_trace, nburn, rho)
#laplacian_diagnostic()
#pack = initial_scan(psi_hydrogen, rhos, x0, nsteps, stepsize, nburn, h_lap=h, seed=1234, doplot=True)