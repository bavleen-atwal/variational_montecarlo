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
    full_x = []
    accepted = 0
    accepted_x = []

    x = np.array(x0, dtype=float)
    logP_x = float(logP_h2(x, rhos, bond_length))

    full_x.append(x.copy())

    for i in range(nsteps):
        x_trial = x + rng.normal(0.0, stepsize, size=6)

        logP_trial = float(logP_h2(x_trial, rhos, bond_length))

        log_r = logP_trial - logP_x
        log_u = np.log(rng.random())

        if (log_r >= 0.0) or (log_u < log_r):
            x = x_trial
            logP_x = logP_trial
            accepted += 1

        full_x.append(x.copy())
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

    return float(kinetic + V_en + V_ee + V_nn)

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
    rhos_min=(1e-6, 1e-6, 1e-6),
    seed=1234,
    verbose=True,
):
    """
    Multivariable gradient-descent optimiser for H2
    """
    rng = np.random.default_rng(seed)

    rhos = np.asarray(rhos0, dtype=float).copy()
    rhos_min = np.asarray(rhos_min, dtype=float)

    rhos_best = rhos.copy()
    E_best = np.inf
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

        if E_mean < E_best:
            E_best = E_mean
            rhos_best = rhos.copy()

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

    return rhos_best, float(E_best), history

def plot_density_log(samples, bond_length, bins=100):
    r1 = samples[:, :3]
    x = r1[:, 0]
    y = r1[:, 1]

    plt.hist2d(x, y, bins=bins, density=True, norm=LogNorm(), cmap='magma')
    #plt.colorbar(label="Probability density")
    plt.xlabel("x (a.u.)")
    plt.ylabel("y (a.u.)")
    plt.title(f"2D projection of electron density for atomic distance {bond_length} a.u.")
    plt.gca().set_facecolor("black")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("equal")
    plt.show()

def plot_density(samples, bond_length, bins=80):
    r1 = samples[:, :3]
    x = r1[:, 0]
    y = r1[:, 1]

    plt.hist2d(x, y, bins=bins, density=True, cmap='viridis')
    plt.colorbar(label="Probability density")
    plt.xlabel("x (a.u.)")
    plt.ylabel("y (a.u.)")
    plt.title(f"2D projection of electron density for atomic distance {bond_length} a.u.")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.axis("equal")
    plt.show()

# def metropolis_3d(logP_h2, x0, rhos, bond_length, nsteps, stepsize, nburn, seed=None):

# Testing
x0_6d = np.array([0.5, 0.0, 0.0,   -0.5, 0.0, 0.0]) 
rhos0 = np.array([1.0, 0.2, 0.5])
bond_length = 1.4
nsteps = 100000
stepsize = 1.0
nburn = 6000
seed = 1234

x_samples, testing_rate = metropolis_3d(
    logP_h2, x0_6d, rhos0, bond_length,
    nsteps, stepsize, nburn, seed)

plot_density(x_samples, bond_length)

"""
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