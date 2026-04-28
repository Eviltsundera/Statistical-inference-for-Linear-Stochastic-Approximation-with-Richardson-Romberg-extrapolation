"""LSA problem generation: A(x), b(x), theta_star."""

import numpy as np

_A_NORM_CAP = 1.0
_A_BAR_EIG_MIN = 0.25
_A_BAR_EIG_MAX = 0.60
_NOISE_NORM_TARGET = 0.35


def _spectral_norms(mats):
    """Return spectral norms for a list/array of matrices."""
    return np.array([np.linalg.norm(mat, ord=2) for mat in mats], dtype=float)


def _generate_hurwitz_mean(d, rng, eig_min, eig_max):
    """Generate a well-conditioned Hurwitz mean matrix.

    The mean matrix is symmetric negative definite with eigenvalues in
    [eig_min, eig_max]. Smaller eig_min → worse conditioning → larger
    bias constant.
    """
    q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    eigvals = rng.uniform(eig_min, eig_max, size=d)
    return -(q @ np.diag(eigvals) @ q.T)


def generate_A(n_states, d, pi, rng,
               eig_min=_A_BAR_EIG_MIN, eig_max=_A_BAR_EIG_MAX,
               noise_target=_NOISE_NORM_TARGET, a_norm_cap=_A_NORM_CAP):
    """Generate state-dependent matrices A(x) with Hurwitz mean.

    Args:
        eig_min, eig_max: range for eigenvalues of -A_bar.
        noise_target: target spectral norm of the state-dependent perturbation.
        a_norm_cap: hard cap on ||A(z)||.

    Returns:
        A_list: list of (d, d) matrices, one per state.
        A_bar: (d, d) mean matrix E_pi[A(x)].
    """
    A_bar = _generate_hurwitz_mean(d, rng, eig_min, eig_max)

    # Generate perturbations for all states and center them under pi.
    # This avoids concentrating the whole correction in a single outlier state.
    raw_noise = rng.uniform(-1, 1, (n_states, d, d))
    centered_noise = raw_noise - np.tensordot(pi, raw_noise, axes=(0, 0))
    centered_noise -= np.tensordot(pi, centered_noise, axes=(0, 0))

    max_noise_norm = float(np.max(_spectral_norms(centered_noise)))
    a_bar_norm = float(np.linalg.norm(A_bar, ord=2))
    noise_budget = max(a_norm_cap - a_bar_norm, 0.0)

    if max_noise_norm > 0 and noise_budget > 0:
        noise_scale = min(
            noise_target / max_noise_norm,
            noise_budget / max_noise_norm,
        )
        centered_noise *= noise_scale
    else:
        centered_noise.fill(0.0)

    A_arr = A_bar[None, :, :] + centered_noise
    return [A_arr[x] for x in range(n_states)], A_bar


def generate_b(n_states, d, rng):
    """Generate state-dependent vectors b(x).

    Returns:
        b_list: list of (d,) vectors, one per state.
    """
    return [rng.uniform(-1, 1, d) for _ in range(n_states)]


def compute_theta_star(A_list, b_list, pi):
    """Compute the true solution theta* = -A_bar^{-1} b_bar.

    Returns:
        theta_star: (d,) true solution vector.
    """
    n_states = len(A_list)
    A_bar = sum(pi[x] * A_list[x] for x in range(n_states))
    b_bar = sum(pi[x] * b_list[x] for x in range(n_states))
    return np.linalg.solve(A_bar, -b_bar)


def compute_asymptotic_variance(A_list, b_list, P, pi, theta_star, direction):
    """Long-run variance along `direction` for PR-averaged LSA with Markov noise.

    Returns sigma^2_inf(u) = u^T Sigma_inf u, where
        Sigma_inf  = A_bar^{-1} Gamma_eps A_bar^{-T},
        Gamma_eps  = E^T (D Z + Z^T D - D) E,
        Z          = (I - P + 1 pi^T)^{-1}        (fundamental matrix),
        D          = diag(pi),
        E          rows = eps(x) = A(x) theta* + b(x).

    Centering of `eps` under `pi` follows from A_bar theta* + b_bar = 0.
    """
    n_states = len(A_list)
    A_bar = sum(pi[x] * A_list[x] for x in range(n_states))

    eps = np.stack(
        [A_list[x] @ theta_star + b_list[x] for x in range(n_states)]
    )
    Pi = np.outer(np.ones(n_states), pi)
    Z = np.linalg.inv(np.eye(n_states) - P + Pi)
    D = np.diag(pi)
    M = D @ Z + Z.T @ D - D
    Gamma = eps.T @ M @ eps

    A_inv = np.linalg.inv(A_bar)
    Sigma = A_inv @ Gamma @ A_inv.T
    return float(direction @ Sigma @ direction)


def problem_diagnostics(A_list, alpha_warn=0.2):
    """Summarize per-problem boundedness and one-step stability diagnostics."""
    d = A_list[0].shape[0]
    eye = np.eye(d)
    max_norm = float(np.max(_spectral_norms(A_list)))
    max_rho = 0.0
    for A in A_list:
        rho = float(np.max(np.abs(np.linalg.eigvals(eye + alpha_warn * A))))
        max_rho = max(max_rho, rho)

    return {
        'alpha_warn': float(alpha_warn),
        'max_a_norm': max_norm,
        'max_rho': max_rho,
        'warn_unstable': bool(max_rho >= 1.0),
        'warn_assumption': bool(max_norm > _A_NORM_CAP + 1e-10),
    }
