import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def nuclear_positions(bond_length):
    """Return posistions of nuclei centred on x axis symmetrically"""
    R = float(bond_length)
    qA = np.array([-0.5 * R, 0.0, 0.0], dtype =float)
    qB = np.array([0.5 * R, 0.0 , 0.0], dtype=float)
    return qA, qB

def safenorm(v, eps=1e-12):
    return max(float(np.linalg.norm(v)), eps)

def psi_h2(V, rhos, bond_length):
    q1, q2 = nuclear_positions(bond_length)
    rho1, rho2, rho3 = np.asarray(rhos, dtype=float)
    V = np.asarray(V, dtype=float)
    r1 = V[:3]
    r2 = V[3:]
    r12 = safenorm(r1 - r2)

    rq11 = safenorm(r1 - q1)
    rq22 = safenorm(r2 - q2)
    rq12 = safenorm(r1 - q2)
    rq21 = safenorm(r2-q1)
    a = -rho1 * (rq11 + rq22)
    b = -rho1 * (rq12 + rq21)
    c = -rho2 / (1 + rho3*r12)
    return (np.exp(a)+np.exp(b)) * np.exp(c)

def logP_h2(V, rhos, bond_length):
    q1, q2 = nuclear_positions(bond_length)
    rho1, rho2, rho3 = np.asarray(rhos, dtype=float)
    V = np.asarray(V, dtype=float)
    r1 = V[:3]
    r2 = V[3:]
    r12 = safenorm(r1 - r2)

    rq11 = safenorm(r1 - q1)
    rq22 = safenorm(r2 - q2)
    rq12 = safenorm(r1 - q2)
    rq21 = safenorm(r2-q1)
    a = -rho1 * (rq11 + rq22)
    b = -rho1 * (rq12 + rq21)
    log2 = -rho2 / (1 + rho3*r12)
    log1 = np.logaddexp(a, b)
    return 2 * (log1 + log2)

def metropolis_3d(logP_h2, x0, rhos, bond_length, nsteps, stepsize, nburn, seed=None):
    """
    6D Metropolis sampler for H2 still using symmetrical gaussian proposals """
    rng = np.random.default_rng(seed)
    accepted = 0
    accepted_x = []

    x = np.array(x0, dtype=float)
    logP_x = float(logP_h2(x, rhos, bond_length))


    for i in range(nsteps):
        x_trial = x + rng.normal(0.0, stepsize, size=6)

        logP_trial = float(logP_h2(x_trial, rhos, bond_length))

        log_r = logP_trial - logP_x
        log_u = np.log(rng.random())

        if (log_r >= 0.0) or (log_u < log_r):
            x = x_trial
            logP_x = logP_trial
            accepted += 1

        if i>= nburn:
            accepted_x.append(x.copy())

        else:
            pass

    acceptance_rate = accepted / nsteps
    return np.asarray(accepted_x, dtype=float) , float(acceptance_rate) 

def laplacian_central(psi, v, rhos, bond_length, h):
    """Laplacian of wavefunction using central difference"""
    v = np.asarray(v, dtype=float)
    psi0= psi(v, rhos, bond_length)
    lap = 0.0
    for i in range(6):
        e = np.zeros(6)
        e[i] = 1.0
        v_plus = v + h * e
        v_minus = v - h * e
        lap += (psi(v_plus, rhos, bond_length) - 2 * psi0 + psi(v_minus, rhos, bond_length)) / (h*h)

    return float(lap)

def local_energy_h2(psi_h2, V, rhos, bond_length, h, r_eps=1e-12, psi_eps=1e-300):
    """ H2 local energy using the Hamiltonian """
    V = np.asarray(V, dtype=float)
    q1, q2 = nuclear_positions(bond_length)

    r1 = V[:3]
    r2 = V[3:]

    # Distances
    r1q1 = safenorm(r1 - q1, eps=r_eps)
    r1q2 = safenorm(r1 - q2, eps=r_eps)
    r2q1 = safenorm(r2 - q1, eps=r_eps)
    r2q2 = safenorm(r2 - q2, eps=r_eps)
    r12  = safenorm(r1 - r2, eps=r_eps)
    q12  = safenorm(q1 - q2, eps=r_eps)

    psi0 = float(psi_h2(V, rhos, bond_length))
    psi0 = max(abs(psi0), psi_eps)

    lap = laplacian_central(psi_h2, V, rhos, bond_length, h)
    kinetic = -0.5 * (lap / psi0)

    V_en = -((1.0 / r1q1) + (1.0 / r1q2) + (1.0 / r2q1) + (1.0 / r2q2))
    V_ee = +1.0 / r12
    V_nn = +1.0 / q12

    return {
        "T": float(kinetic),
        "V_en": float(V_en),
        "V_ee": float(V_ee),
        "V_nn": float(V_nn),
        "E_total": float(kinetic + V_en + V_ee + V_nn),
        "q12": float(q12),  # for sanity: should be ~bond_length
        "r12": float(r12),
    }

def diagnose_energy_terms_at_R(
    psi_h2, logP_h2, rhos, bond_length,
    x0_6d, nsteps, stepsize, nburn, h,
    seed=0, max_print=5
):
    """
    Runs a sampling pass at fixed rhos and prints mean ± std for each term.
    Also prints a few individual samples so you can spot crazy outliers.
    """
    # Use your existing sampler (your metropolis function that returns accepted samples + acc)
    samples, acc = metropolis_3d(  # rename if yours differs
        logP_h2=logP_h2,
        x0=x0_6d,
        rhos=rhos,
        bond_length=bond_length,
        nsteps=nsteps,
        stepsize=stepsize,
        nburn=nburn,
        seed=seed
    )

    terms_list = []
    for V in samples:
        terms_list.append(local_energy_h2(psi_h2, V, rhos, bond_length, h))

    # Collect arrays
    T    = np.array([d["T"] for d in terms_list])
    Ven  = np.array([d["V_en"] for d in terms_list])
    Vee  = np.array([d["V_ee"] for d in terms_list])
    Vnn  = np.array([d["V_nn"] for d in terms_list])
    Etot = np.array([d["E_total"] for d in terms_list])
    q12s = np.array([d["q12"] for d in terms_list])

    def mean_std(x):
        return float(np.mean(x)), float(np.std(x, ddof=1))

    print(f"\n--- Term breakdown at R={bond_length} ---")
    print(f"Acceptance: {acc:.3f}")
    print(f"Mean q12 (should be ~R): {np.mean(q12s):.6f}  (min={np.min(q12s):.6f}, max={np.max(q12s):.6f})")

    for name, arr in [("T", T), ("V_en", Ven), ("V_ee", Vee), ("V_nn", Vnn), ("E_total", Etot)]:
        m, s = mean_std(arr)
        print(f"{name:7s}: mean={m:+.6f}   std={s:.6f}")

    # Sanity: V_nn should be basically constant = 1/R
    print(f"Expected V_nn = 1/R = {1.0/bond_length:.6f}")

    # Print a few individual samples (useful for catching sign explosions/outliers)
    print("\nExample individual configurations (first few):")
    for i in range(min(max_print, len(terms_list))):
        d = terms_list[i]
        print(f"[{i:02d}] E={d['E_total']:+.6f}  T={d['T']:+.6f}  V_en={d['V_en']:+.6f}  V_ee={d['V_ee']:+.6f}  V_nn={d['V_nn']:+.6f}  r12={d['r12']:.4f}")

    return {
        "acc": acc,
        "T": T, "V_en": Ven, "V_ee": Vee, "V_nn": Vnn, "E_total": Etot
    }
R = 5
rhos_test = np.array([1.4279984, 0.19608351, 0.50236534], dtype=float)

x0_6d = np.array([0.5, 0.0, 0.0,  -0.5, 0.0, 0.0], dtype=float)

diag = diagnose_energy_terms_at_R(
    psi_h2=psi_h2,
    logP_h2=logP_h2,
    rhos=rhos_test,
    bond_length=R,
    x0_6d=x0_6d,
    nsteps=30000,     # start moderate; you can increase later
    stepsize=0.5,     # use your “good acceptance” step
    nburn=3000,
    h=1e-4,
    seed=123
)
def grad_logpsi_h2(V, rhos, bond_length):
    rho1, rho2, rho3 = rhos
    q1, q2 = nuclear_positions(bond_length)

    r1 = V[:3]
    r2 = V[3:]
    r12 = safenorm(r1 - r2)

    r1q1 = safenorm(r1 - q1)
    r1q2 = safenorm(r1 - q2)
    r2q1 = safenorm(r2 - q1)
    r2q2 = safenorm(r2 - q2)

    Sa = r1q1 + r2q2
    Sb = r1q2 + r2q1

    a = -rho1 * Sa
    b = -rho1 * Sb

    # weights w_a, w_b
    m = max(a, b) # preventing underflow errors for very small a  and b
    ea = np.exp(a - m)
    eb = np.exp(b - m)
    wa = ea / (ea + eb)
    wb = eb / (ea + eb)

    d_rho1 = -(wa * Sa + wb * Sb)
    d_rho2 = -(1.0 / (1.0 + rho3 * r12))
    d_rho3 = (rho2 * r12) / (1.0 + rho3 * r12)**2

    return np.array([d_rho1, d_rho2, d_rho3])

def gradient_descent_minimise_rhos_h2(
    psi_h2,
    logP_h2,
    rhos0,
    x0,
    bond_length,
    nsteps,
    stepsize,
    nburn,
    h,
    lr=0.02,
    max_iter=40,
    tol_grad=1e-3,
    tol_rhos=1e-4,
    rhos_min=(1e-7, 1e-7, 1e-7),
    seed=1234,
    verbose=True,
):
    """
    Multivariable gradient-descent optimiser for H2
    """
    rng = np.random.default_rng(seed)

    rhos = np.asarray(rhos0, dtype=float).copy()
    rhos_min = np.asarray(rhos_min, dtype=float)

    history = []

    for it in range(1, max_iter + 1):
        seed_it = int(rng.integers(0, 2**31 - 1))

        samples, acc = metropolis_3d(
            logP_h2=logP_h2,
            x0=x0,
            rhos=rhos,
            bond_length=bond_length,
            nsteps=nsteps,
            stepsize=stepsize,
            nburn=nburn,
            seed=seed_it
        )

        E_samples = np.array(
            [local_energy_h2(psi_h2, V, rhos, bond_length, h) for V in samples],
            dtype=float)
        E_mean = float(np.mean(E_samples))
        E_std = float(np.std(E_samples, ddof=1))
        E_sem = E_std / np.sqrt(len(E_samples))

        # Gradient estimate
        grads = np.zeros(3, dtype=float)
        for Ei, Vi in zip(E_samples, samples):
            grads += (Ei - E_mean) * grad_logpsi_h2(Vi, rhos, bond_length)
        grads *= (2.0 / len(E_samples))

        grad_norm = float(np.linalg.norm(grads))

        # Gradient step
        rhos_new = rhos - lr * grads

        # Enforce positive rhos
        rhos_new = np.maximum(rhos_new, rhos_min)

        history.append({
            "iter": it,
            "rhos": rhos.copy(),
            "E_mean": E_mean,
            "E_sem": E_sem,
            "acc": float(acc),
            "grads": grads.copy(),
            "grad_norm": grad_norm,
            "rhos_new": rhos_new.copy(),
        })

        if verbose:
            print(
                f"[{it:02d}] rhos={rhos}  E={E_mean:.6f} ± {E_sem:.6f}  "
                f"|grad|={grad_norm:.3e}  acc={acc:.3f}  -> rhos_new={rhos_new}")

        # Stopping conditions
        if grad_norm < tol_grad:
            if verbose:
                print(f"Stop: ||grad|| < tol_grad ({tol_grad:g}).")
            rhos = rhos_new
            break

        if float(np.linalg.norm(rhos_new - rhos)) < tol_rhos:
            if verbose:
                print(f"Stop: ||Δrhos|| < tol_rhos ({tol_rhos:g}).")
            rhos = rhos_new
            break

        rhos = rhos_new

    return rhos, float(E_mean), history

def estimate_energy_h2_fixed_rhos(psi_h2, logP_h2, rhos, x0, bond_length,
                                 nsteps, stepsize, nburn, h, seed=0):
    """Sample with fixed rhos and return E mean ± SEM + acceptance + samples for final energy"""
    samples, acc = metropolis_3d(
        logP_h2=logP_h2, x0=x0, rhos=rhos, bond_length=bond_length,
        nsteps=nsteps, stepsize=stepsize, nburn=nburn, seed=seed
    )

    E_samples = np.array([local_energy_h2(psi_h2, V, rhos, bond_length, h)
                          for V in samples], dtype=float)

    E_mean = float(np.mean(E_samples))
    E_std  = float(np.std(E_samples, ddof=1))
    E_sem  = float(E_std / np.sqrt(len(E_samples)))

    return E_mean, E_sem, float(acc), samples

def morse_potential(r, D, a, r0, E_single):
    return D * (1.0 - np.exp(-a * (r - r0)))**2 - D + 2.0 * E_single

def plot_binding_curve(results):
    R = np.array([d["R"] for d in results], dtype=float)
    E = np.array([d["E"] for d in results], dtype=float)
    dE = np.array([d["E_sem"] for d in results], dtype=float)

    plt.figure(figsize=(6,4))
    plt.errorbar(R, E, yerr=dE, fmt="o", capsize=3)
    plt.xlabel("Bond length R (a.u.)")
    plt.ylabel("Ground state energy E(R)")
    plt.title("H₂ binding curve from VMC")
    plt.tight_layout()
    plt.show()

def fit_morse(results, E_single=-0.5):
    R = np.array([d["R"] for d in results], dtype=float)
    E = np.array([d["E"] for d in results], dtype=float)

    # initial guesses: r0 ~ 1.4, D ~ 0.17, a ~ 1
    p0 = np.array([0.17, 1.0, 1.4], dtype=float)

    try:
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(
            lambda r, D, a, r0: morse_potential(r, D, a, r0, E_single),
            R, E, p0=p0, maxfev=20000
        )
        perr = np.sqrt(np.diag(pcov))
    except Exception as e:
        print("curve_fit failed, reason:", e)
        popt, perr = p0, np.array([np.nan, np.nan, np.nan])

    D, a, r0 = popt
    print(f"Morse fit: D={D:.5f}, a={a:.5f}, r0={r0:.5f}  (E_single={E_single:.5f})")

    # plot fit
    rgrid = np.linspace(R.min(), R.max(), 300)
    Efit = morse_potential(rgrid, D, a, r0, E_single)

    plt.figure(figsize=(6,4))
    plt.plot(R, E, "o", label="VMC")
    plt.plot(rgrid, Efit, "-", label="Morse fit")
    plt.xlabel("Bond length R (a.u.)")
    plt.ylabel("Energy E(R)")
    plt.title("Morse fit to VMC binding curve")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return popt, perr

def random_initial_rhos(rng, rho1=(0.6, 1.4), rho2=(0.05, 0.5), rho3=(0.1, 2.0)):
    return np.array([
        rng.uniform(*rho1),
        rng.uniform(*rho2),
        rng.uniform(*rho3),
    ], dtype=float)

def plot_density(samples, bond_length, bins=100, use_log=False):
    """
    Projected electron density: add BOTH electrons to the 2D histogram.
    """
    samples = np.asarray(samples, dtype=float)
    r1 = samples[:, :3]
    r2 = samples[:, 3:]

    x = np.concatenate([r1[:, 0], r2[:, 0]])
    y = np.concatenate([r1[:, 1], r2[:, 1]])

    plt.figure(figsize=(6, 4.8))
    if use_log:
        plt.hist2d(
            x, y, bins=bins,
            density=True, cmap="magma",
            norm=LogNorm(vmin=1e-3)
        )
    else:
        plt.hist2d(
            x, y, bins=bins,
            density=True, cmap="viridis"
        )

    plt.colorbar(label="Probability density")
    plt.xlabel("x (a.u.)")
    plt.ylabel("y (a.u.)")
    plt.title(f"Projected electron density (both electrons), R={bond_length:.2f}")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.show()


# Running all code

def run_h2_bond_scan(
    bond_lengths,
    rhos_init,
    x0_6d,
    # optimiser settings
    opt_nsteps=50000,
    opt_stepsize=0.5,
    opt_nburn=6000,
    h=1e-4,
    lr=0.02,
    opt_max_iter=70,
    # final energy estimate settings
    eval_nsteps=120000,
    eval_stepsize=0.5,
    eval_nburn=8000,
    seed=1234,
    make_density_plots=True
):
    """For each bond length R:
      1) optimise rhos
      2) re-sample with rhos fixed to estimate energy
      3) plot density (both electrons)
    Returns arrays for E(R) and fitted parameters later.
    """
    results = []
    rhos0 = np.asarray(rhos_init, dtype=float)

    for k, R in enumerate(bond_lengths):
        print(f"\n=== Bond length R={R:.3f} ===")

        rhos_opt, E_opt, hist = gradient_descent_minimise_rhos_h2(
            psi_h2=psi_h2,
            logP_h2=logP_h2,
            rhos0=rhos0,
            x0=x0_6d,
            bond_length=R,
            nsteps=opt_nsteps,
            stepsize=opt_stepsize,
            nburn=opt_nburn,
            h=h,
            lr=lr,
            max_iter=opt_max_iter,
            verbose=True,
            seed=seed + 1000*k
        )

        # 2) energy estimate with rhos fixed (clean report value)
        E_mean, E_sem, acc_eval, samples = estimate_energy_h2_fixed_rhos(
            psi_h2, logP_h2, rhos_opt, x0_6d, R,
            nsteps=eval_nsteps, stepsize=eval_stepsize, nburn=eval_nburn,
            h=h, seed=seed + 2000*k
        )

        print(f"FINAL (fixed rhos): R={R:.3f}  rhos={rhos_opt}  "
              f"E={E_mean:.6f} ± {E_sem:.6f}  acc={acc_eval:.3f}")

        # 3) required density plot
        if make_density_plots:
            plot_density(samples, R, bins=100, use_log=False)

        results.append({
            "R": float(R),
            "rhos": np.array(rhos_opt, dtype=float),
            "E": float(E_mean),
            "E_sem": float(E_sem),
            "acc": float(acc_eval),
        })

        # warm start next R
        rhos0 = rhos_opt.copy()

    return results

"""
x0_6d = np.array([0.5, 0.0, 0.0,   -0.5, 0.0, 0.0]) 
rhos0 = np.array([1.0, 0.2, 0.5])
bond_lengths = np.linspace(0.5, 3.0, 5)
bond_length = 2
nsteps = 100000
stepsize = 0.5
nburn = 6000
seed = 1234


results = run_h2_bond_scan(
        bond_lengths=bond_lengths,
        rhos_init=rhos0,
        x0_6d=x0_6d,
        opt_stepsize=0.5,     # acceptance too low? keep < 1 for now
        eval_stepsize=0.5,
        make_density_plots=True
    )

plot_binding_curve(results)
popt, perr = fit_morse(results, E_single=-0.5)
"""
"""
x_samples, testing_rate = metropolis_3d(
    logP_h2, x0_6d, rhos0, bond_length,
    nsteps, stepsize, nburn, seed)

plot_density(x_samples, bond_length)
rhos_opt, E_opt, hist = gradient_descent_minimise_rhos_h2(
    psi_h2=psi_h2,
    logP_h2=logP_h2,
    rhos0=rhos0,
    x0=x0_6d,
    bond_length=bond_length,
    nsteps=80000,
    stepsize=1.0,
    nburn=6000,
    h=1e-4,
    lr=0.02,
    max_iter=80,
    verbose=True
)

print("Best rhos:", rhos_opt, "Best E:", E_opt)
"""