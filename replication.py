"""
Replication of Section 7 case study from:
  "Graphon Games" — Parise & Ozdaglar (2019)

Setup (Table 1 / Figure 5):
  - K=4 community SBM graphon (tridiagonal structure)
  - intra-community probs: [0.9, 0.2, 0.2, 0.8]
  - adjacent inter-community prob: 0.05 (only C1<->C2, C2<->C3, C3<->C4)
  - non-adjacent pairs (C1<->C3, C1<->C4, C2<->C4): 0
  - alpha = 2.65
  - theta per community: [0.1, 0.1, 0.1, 0.25]
  - sparsity: kappa_N = 40 / N^0.8
  - Per-agent budget: C = 0.02, constraint: (1/N)||theta_hat||^2 <= C
  - M = 80 network samples per N

Budget constraint (L2): (1/N)||theta_hat||^2 <= C  =>  ||theta_hat||^2 <= C*N

Three interventions compared (all vs. homogeneous baseline):
  1. Network optimal   -- eigendecomposition + brentq on full N x N system
  2. Network heuristic -- theta_hat proportional to dominant eigenvector of W_N
  3. Graphon optimal   -- eigendecomposition + brentq on K x K system, then project
"""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq

# 1. Graphon setup (SBM with K=4 communities)

K = 4
ALPHA = 2.65
THETA_COM = np.array([0.1, 0.1, 0.1, 0.25])

# Per-agent budget parameter (paper notation: C = 0.02)
# Constraint: (1/N) * sum_i theta_hat_i^2 <= C
# Equivalently: sum_i theta_hat_i^2 <= C * N
C_PER_AGENT = 0.02

# SBM connectivity matrix Q (K×K) — TRIDIAGONAL
# Only adjacent communities connected: C1<->C2, C2<->C3, C3<->C4 at prob 0.05
# Non-adjacent pairs C1<->C3, C1<->C4, C2<->C4: prob 0
Q_SBM = np.zeros((K, K))
np.fill_diagonal(Q_SBM, [0.9, 0.2, 0.2, 0.8])
for c in range(K - 1):
    Q_SBM[c, c + 1] = 0.05
    Q_SBM[c + 1, c] = 0.05

KAPPA_COEF = 40.0  # kappa_N = 40 / N^0.8


def get_kappa(N):
    return KAPPA_COEF / (N ** 0.8)


# 2. Graphon equilibrium (K×K system)

def graphon_equilibrium(alpha=ALPHA, theta_com=THETA_COM, q_sbm=Q_SBM, k=K):
    """
    Solve the LQ graphon equilibrium on the K-community SBM.

    Equal community sizes pi_c = 1/K, so the auxiliary matrix is:
        E[c,c'] = Q[c,c'] * pi_c' = Q[c,c'] / K

    Equilibrium (Eq. 18):
        s_bar = (I - alpha * E)^{-1} * theta_com
    """
    E = q_sbm / k
    s_bar = np.linalg.solve(np.eye(k) - alpha * E, theta_com)
    return np.maximum(s_bar, 0.0)


def graphon_welfare(s_bar, alpha=ALPHA, theta_com=THETA_COM, q_sbm=Q_SBM, k=K):
    """
    Aggregate welfare under graphon equilibrium.
    At equilibrium, welfare = (1/2) * sum_c pi_c * s_c^2 = (1/2K) sum_c s_c^2.
    """
    return 0.5 * np.mean(s_bar**2)


# 3. Network sampling

def sample_network(N, rng, q_sbm=Q_SBM, k=K):
    """
    Sample one network of size N from the SBM graphon.

    Types t^i drawn uniformly from [0,1]; community c = floor(t * K).
    Edge (i,j) drawn with Bernoulli(kappa_N * Q[c_i, c_j]).

    Returns: A, community, theta, kappa
    """
    kappa = get_kappa(N)

    t = rng.uniform(0, 1, size=N)
    community = np.clip((t * k).astype(int), 0, k - 1)

    ci = community[:, None]
    cj = community[None, :]
    prob = np.clip(kappa * q_sbm[ci, cj], 0.0, 1.0)

    U = rng.uniform(0, 1, size=(N, N))
    A = (U < prob).astype(float)
    A = np.triu(A, 1)
    A = A + A.T

    theta = THETA_COM[community]
    return A, community, theta, kappa


# 4. Network equilibrium  s* = (I - alpha * W_N)^{-1} theta
#    W_N = A / (kappa_N * N)   [sparse normalization, Section 5.1 Def 9]

def network_equilibrium(A, theta, alpha=ALPHA, kappa=None):
    """
    LQ Nash equilibrium on finite network.
    W_N = A / (kappa_N * N)
    s* = (I - alpha * W_N)^{-1} * theta
    """
    N = A.shape[0]
    if kappa is None:
        kappa = get_kappa(N)
    W_N = A / (kappa * N)
    M = np.eye(N) - alpha * W_N
    try:
        s_star = np.linalg.solve(M, theta)
    except np.linalg.LinAlgError:
        s_star = np.linalg.lstsq(M, theta, rcond=None)[0]
    return np.maximum(s_star, 0.0)


def network_welfare(s):
    """
    Welfare at equilibrium: T = (1/2N) * ||s||^2.

    This uses the LQ equilibrium identity: at NE, each agent's payoff
    U_i = -1/2 s_i^2 + s_i*(alpha*z_i + theta_i + theta_hat_i) = 1/2 s_i^2.

    This form is correct regardless of what theta was used to compute s,
    so we don't need to track theta vs theta+theta_hat.
    """
    return 0.5 * np.mean(s**2)


# 5. Core eigendecomposition solver (Problems 14 & 16)
#
#   max_{||eta||^2 <= budget}  ||(I - alpha*W)^{-1} * (theta + eta)||^2
#
#   Diagonalize W = V D V^T.  Let m_j = 1/(1 - alpha*d_j).
#   In eigenbasis:  objective = sum_j m_j^2 * (xi_j + eta_j)^2,  xi = V^T theta.
#   KKT with Lagrange multiplier mu on ||eta||^2 <= budget:
#       eta_j = m_j^2 * xi_j / (mu - m_j^2)
#   Find mu > max(m_j^2) via brentq so ||eta||^2 = budget.

def _optimal_subsidy(W, theta, budget, alpha=ALPHA):
    """
    Solve the constrained QP over a ball (Problems 14/16).
    Returns theta_hat (optimal subsidy vector, same dimension as theta).
    """
    eigvals_W, V = eigh(W)                          # ascending order
    m = 1.0 / (1.0 - alpha * eigvals_W)             # eigenvalues of M^{-1}
    m2 = m ** 2

    xi = V.T @ theta                                 # project theta

    def residual(mu):
        eta = m2 * xi / (mu - m2)
        return np.dot(eta, eta) - budget

    mu_lo = m2.max() * (1.0 + 1e-9)

    # Check if budget is so small the constraint doesn't bind
    if residual(mu_lo) <= 0.0:
        g = m2 * xi
        nrm = np.sqrt(np.dot(g, g))
        eta_opt = (np.sqrt(budget) * g / nrm) if nrm > 0 else np.zeros_like(xi)
        return V @ eta_opt

    mu_hi = mu_lo * 2.0
    for _ in range(80):
        if residual(mu_hi) < 0.0:
            break
        mu_hi *= 2.0
    else:
        eta_opt = np.zeros_like(xi)
        idx = np.argmax(m2)
        eta_opt[idx] = np.sqrt(budget) * np.sign(xi[idx]) if xi[idx] != 0 else np.sqrt(budget)
        return V @ eta_opt

    mu_opt = brentq(residual, mu_lo, mu_hi, xtol=1e-12, rtol=1e-12)
    eta_opt = m2 * xi / (mu_opt - m2)
    return V @ eta_opt


# 6. Interventions

def intervention_homogeneous(A, theta, alpha=ALPHA, C=C_PER_AGENT, kappa=None):
    """
    Homogeneous baseline: equal subsidy to all agents.
    (1/N)||theta_hat||^2 = C  =>  theta_hat_i = sqrt(C) for all i.
    """
    N = A.shape[0]
    theta_hat = np.full(N, np.sqrt(C))
    return network_equilibrium(A, theta + theta_hat, alpha, kappa=kappa)


def intervention_network_optimal(A, theta, alpha=ALPHA, C=C_PER_AGENT, kappa=None):
    """
    Network-optimal (Problem 14):
        max  (1/2N) ||s*(theta + theta_hat)||^2
        s.t. (1/N) ||theta_hat||^2 <= C   <=>   ||theta_hat||^2 <= C*N
    """
    N = A.shape[0]
    if kappa is None:
        kappa = get_kappa(N)
    W_N = A / (kappa * N)
    theta_hat = _optimal_subsidy(W_N, theta, budget=C * N, alpha=alpha)
    return network_equilibrium(A, theta + theta_hat, alpha, kappa=kappa)


def intervention_network_heuristic(A, theta, alpha=ALPHA, C=C_PER_AGENT, kappa=None):
    """
    Network heuristic:
        theta_hat = c * v1,  v1 = dominant eigenvector of W_N (unit norm)
        c = sqrt(C*N)  so that (1/N)||theta_hat||^2 = C.

    Sign: ensure v1 points in the positive direction (Perron-Frobenius).
    """
    N = A.shape[0]
    if kappa is None:
        kappa = get_kappa(N)
    W_N = A / (kappa * N)
    _, eigvecs = eigh(W_N)
    v1 = eigvecs[:, -1]                               # unit-norm

    # Enforce nonneg direction (PF theorem: dominant eigvec of nonneg matrix is nonneg)
    if np.sum(v1) < 0:
        v1 = -v1

    # Scale: ||theta_hat||^2 = C*N,  theta_hat = c*v1,  ||v1||=1  =>  c = sqrt(C*N)
    theta_hat = np.sqrt(C * N) * v1
    return network_equilibrium(A, theta + theta_hat, alpha, kappa=kappa)


def intervention_graphon_optimal(A, community, theta, alpha=ALPHA, C=C_PER_AGENT,
                                  kappa=None, q_sbm=Q_SBM, k=K):
    """
    Graphon-optimal (Problem 16): solve K x K optimization.

    For equal community sizes pi_c = 1/K, the L2 budget is:
        ||theta_hat||^2_{L2} = (1/K) sum_c delta_c^2 <= C
        =>  sum_c delta_c^2 <= K * C

    Then agent i in community c gets theta_hat_i = delta_c.
    A normalization factor eta ensures the per-network budget is met
    exactly: (1/N) sum_i theta_hat_i^2 = C  (Theorem 6).
    """
    N = A.shape[0]
    if kappa is None:
        kappa = get_kappa(N)

    E = q_sbm / k
    delta = _optimal_subsidy(E, THETA_COM, budget=k * C, alpha=alpha)

    # Map to agents and renormalize to meet per-network budget exactly
    theta_hat = delta[community]
    actual_budget = np.sum(theta_hat**2) / N
    if actual_budget > 0:
        theta_hat *= np.sqrt(C / actual_budget)

    return network_equilibrium(A, theta + theta_hat, alpha, kappa=kappa)


# 7. Main experiment (Table 1 replication)

def run_experiment(N_list=(300, 600, 1200), M=80, seed=42):
    """
    Replicate Table 1: welfare improvement (%) over homogeneous baseline.

    improvement_g = welfare(intervention_g) / welfare(homogeneous) - 1
    averaged over M network draws.
    """
    results = {}

    for N in N_list:
        kN = get_kappa(N)
        print(f"\n=== N = {N}  (kappa_N = {kN:.4f}) ===")
        rng = np.random.default_rng(seed + N)

        w_hom_list = []
        w_net_opt_list = []
        w_net_heur_list = []
        w_graphon_opt_list = []
        deg_all = []
        deg_com = {c: [] for c in range(K)}

        for m in range(M):
            A, community, theta, kappa = sample_network(N, rng)

            # Compute equilibria under each intervention
            s_h  = intervention_homogeneous(A, theta, kappa=kappa)
            s_no = intervention_network_optimal(A, theta, kappa=kappa)
            s_nh = intervention_network_heuristic(A, theta, kappa=kappa)
            s_go = intervention_graphon_optimal(A, community, theta, kappa=kappa)

            # Welfare = (1/2N) ||s||^2
            w_hom_list.append(network_welfare(s_h))
            w_net_opt_list.append(network_welfare(s_no))
            w_net_heur_list.append(network_welfare(s_nh))
            w_graphon_opt_list.append(network_welfare(s_go))

            # Track degrees
            d = A.sum(axis=1)
            deg_all.append(d.mean())
            for c in range(K):
                deg_com[c].append(d[community == c].mean())

            if (m + 1) % 20 == 0:
                print(f"  Sample {m+1}/{M} done")

        w_hom = np.array(w_hom_list)

        def pct(w_arr):
            return 100.0 * (np.array(w_arr) - w_hom) / np.abs(w_hom)

        results[N] = {
            'w_hom':           w_hom,
            'pct_net_opt':     pct(w_net_opt_list),
            'pct_net_heur':    pct(w_net_heur_list),
            'pct_graphon_opt': pct(w_graphon_opt_list),
            'deg_all':         np.mean(deg_all),
            'deg_com':         {c: np.mean(deg_com[c]) for c in range(K)},
        }

        dc = results[N]['deg_com']
        deg_str = ', '.join(f'C{c+1}={dc[c]:.1f}' for c in range(K))
        print(f"  Avg degree: {results[N]['deg_all']:.1f}  [{deg_str}]")
        print(f"  Welfare improvement over homogeneous baseline (%):")
        print(f"    Net optimal:    {results[N]['pct_net_opt'].mean():5.1f}%  "
              f"(+/-{results[N]['pct_net_opt'].std():.1f})")
        print(f"    Net heuristic:  {results[N]['pct_net_heur'].mean():5.1f}%  "
              f"(+/-{results[N]['pct_net_heur'].std():.1f})")
        print(f"    Graphon opt:    {results[N]['pct_graphon_opt'].mean():5.1f}%  "
              f"(+/-{results[N]['pct_graphon_opt'].std():.1f})")

    return results


def print_table(results):
    print("\n" + "=" * 70)
    print(f"{'':>6} | {'Net Optimal':>16} | {'Net Heuristic':>16} | {'Graphon Opt':>16}")
    print("-" * 70)
    for N, res in sorted(results.items()):
        def fmt(key):
            return f"{res[key].mean():5.1f}% +/-{res[key].std():4.1f}"
        print(f"N={N:>4} | {fmt('pct_net_opt'):>16} | {fmt('pct_net_heur'):>16} | {fmt('pct_graphon_opt'):>16}")
    print("=" * 70)


# 8. Sanity checks

def print_graphon_eq():
    s_bar = graphon_equilibrium()
    w_bar = graphon_welfare(s_bar)
    E = Q_SBM / K
    lam_max = eigh(E, eigvals_only=True)[-1]
    print("\nGraphon equilibrium (K=4, tridiagonal Q_SBM):")
    print(f"  Q_SBM =\n{Q_SBM}")
    print(f"  E = Q/K =\n{E}")
    for c in range(K):
        print(f"  Community {c+1}: s_bar = {s_bar[c]:.4f}  (theta={THETA_COM[c]})")
    print(f"  Graphon welfare = {w_bar:.6f}")
    print(f"  lambda_max(E) = {lam_max:.4f},  alpha*lambda_max = {ALPHA * lam_max:.4f}  (need <1)")


if __name__ == "__main__":
    print_graphon_eq()
    results = run_experiment(N_list=[300, 600], M=80, seed=42)
    print_table(results)