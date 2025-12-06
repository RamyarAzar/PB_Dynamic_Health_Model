"""
PB-Based Dynamic State–Space Model with Prevention Core (Version 5)

Changes vs v4:
- Fix phase_full construction for CSV and phase plot:
    phase_full = np.concatenate([S_hat, [S_hat[-1]]])
  so that phase[i] is shown exactly on [t_i, t_{i+1}] with plt.step(..., where="post").
- Everything else kept identical to v4.

Usage example: see bottom of file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse


# ---------------------------------------------------------------------------
# Utility: random generator helper
# ---------------------------------------------------------------------------

def get_rng(random_state: Optional[np.random.Generator | int] = None) -> np.random.Generator:
    """
    Convert an integer seed or an existing Generator into a numpy Generator.
    """
    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


# ---------------------------------------------------------------------------
# 1) Observed time series and derived quantities
# ---------------------------------------------------------------------------

@dataclass
class ObservedSeries:
    """
    Observed univariate biomarker or PB risk score {y_n} with times {t_n}.

    Implements:
    - Baseline y_base
    - Normalized series x_n
    - Log-normalized z_n
    - Ratios r_n and log-ratios ℓ_n
    - Smoothed log-ratios ℓ̃_n
    - Early-instability index EI_n
    - Health-basin potential V_single(n) = z_n^2
    - Exit time from the health basin
    """
    y: np.ndarray
    t: np.ndarray
    healthy_indices: np.ndarray
    W: int = 5

    y_base: Optional[float] = field(init=False, default=None)
    x: Optional[np.ndarray] = field(init=False, default=None)
    z: Optional[np.ndarray] = field(init=False, default=None)
    r: Optional[np.ndarray] = field(init=False, default=None)
    ell: Optional[np.ndarray] = field(init=False, default=None)
    ell_smooth: Optional[np.ndarray] = field(init=False, default=None)
    EI: Optional[np.ndarray] = field(init=False, default=None)

    def __post_init__(self):
        self.y = np.asarray(self.y, dtype=float)
        self.t = np.asarray(self.t, dtype=float)
        self.healthy_indices = np.asarray(self.healthy_indices, dtype=int)

        if self.y.shape != self.t.shape:
            raise ValueError("y and t must have the same shape (N+1,).")
        if np.any(self.y <= 0):
            raise ValueError("All y_n must be strictly positive.")
        # monotonic time check
        if not np.all(np.diff(self.t) > 0):
            raise ValueError("t_n must be strictly increasing.")
        if self.W < 1:
            raise ValueError("Window size W must be >= 1.")

    def compute_baseline(self) -> float:
        """
        Baseline y_base = mean over healthy window H.
        """
        idx = self.healthy_indices
        if idx.size == 0:
            raise ValueError("healthy_indices (H) is empty.")
        self.y_base = float(self.y[idx].mean())
        return self.y_base

    def compute_normalized_log(self, c: Optional[float] = None) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        If c is provided (log-baseline), z_n = log y_n - c and x is None.
        Otherwise x_n = y_n / y_base, z_n = log x_n.
        """
        if c is None:
            if self.y_base is None:
                self.compute_baseline()
            x = self.y / self.y_base
            z = np.log(x)
        else:
            x = None
            z = np.log(self.y) - c

        self.x = x
        self.z = z
        return x, z

    def compute_ratios(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Single-step ratios r_n and log-ratios ℓ_n.
        """
        if self.z is None:
            raise RuntimeError("compute_normalized_log() must be called before compute_ratios().")

        z = self.z
        r = np.exp(z[1:] - z[:-1])
        ell = z[1:] - z[:-1]

        self.r = r
        self.ell = ell
        return r, ell

    def compute_smoothed_log_ratios(self) -> np.ndarray:
        """
        Smoothed log-ratio ℓ̃_n using trailing window W.
        """
        if self.ell is None:
            raise RuntimeError("compute_ratios() must be called before compute_smoothed_log_ratios().")

        ell = self.ell
        N = ell.shape[0]
        W = self.W
        ell_smooth = np.empty_like(ell)

        for n in range(N):
            k_start = max(0, n - W + 1)
            window = ell[k_start:n+1]
            ell_smooth[n] = window.mean()

        self.ell_smooth = ell_smooth
        return ell_smooth

    def compute_EI(self, r_band_low: float, r_band_high: float) -> np.ndarray:
        """
        Early-instability index EI_n based on ratios.
        """
        if self.r is None:
            raise RuntimeError("compute_ratios() must be called before compute_EI().")

        r = self.r
        N = r.shape[0]
        W = self.W
        EI = np.empty_like(r)

        for n in range(N):
            k_start = max(0, n - W + 1)
            window = r[k_start:n+1]
            in_band = (window >= r_band_low) & (window <= r_band_high)
            EI[n] = in_band.mean()

        self.EI = EI
        return EI

    def potential_single(self, z: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Single-axis potential V_single(n) = z_n^2.
        """
        if z is None:
            if self.z is None:
                raise RuntimeError("compute_normalized_log() must be called before potential_single().")
            z = self.z
        return z ** 2

    def exit_time_from_health_basin(
        self,
        theta_H: float,
        V_H_crit: Optional[float] = None
    ) -> Optional[float]:
        """
        Exit time from health basin based on |z_n| or V_single.
        """
        if self.z is None:
            raise RuntimeError("compute_normalized_log() must be called before exit_time_from_health_basin().")

        z = self.z
        t = self.t

        if V_H_crit is not None:
            V_single = self.potential_single(z)
            mask = V_single > V_H_crit
        else:
            mask = np.abs(z) > theta_H

        if not np.any(mask):
            return None

        first_idx = np.argmax(mask)
        return float(t[first_idx])


# ---------------------------------------------------------------------------
# 2) PB Phase model as Markov chain with ratio windows
# ---------------------------------------------------------------------------

@dataclass
class PBPhaseModel:
    """
    PB phases S as a discrete Markov chain with transition matrix P and
    phase-specific PB ratio windows and canonical ratios.

    Note:
        - sigma_proc stores the *variances* σ_s^2 (not std).
    """
    states: List[str]
    P: np.ndarray
    r_min: np.ndarray
    r_max: np.ndarray
    r_center: np.ndarray
    sigma_r: np.ndarray
    dt_ref: float = 1.0
    sigma_proc: Optional[np.ndarray] = None
    kappa: Optional[np.ndarray] = None
    z_eq: Optional[np.ndarray] = None

    def __post_init__(self):
        self.P = np.asarray(self.P, dtype=float)
        S = len(self.states)
        if self.P.shape != (S, S):
            raise ValueError("P must have shape (S, S).")

        # Normalize row sums
        row_sums = self.P.sum(axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError("Each row of P must have positive sum.")
        self.P = self.P / row_sums

        self.r_min = np.asarray(self.r_min, dtype=float)
        self.r_max = np.asarray(self.r_max, dtype=float)
        self.r_center = np.asarray(self.r_center, dtype=float)
        self.sigma_r = np.asarray(self.sigma_r, dtype=float)

        for arr_name, arr in [
            ("r_min", self.r_min),
            ("r_max", self.r_max),
            ("r_center", self.r_center),
            ("sigma_r", self.sigma_r),
        ]:
            if arr.shape[0] != S:
                raise ValueError(f"{arr_name} must have shape (S,) with S = len(states).")

        if np.any(self.sigma_r <= 0):
            raise ValueError("All sigma_r must be > 0.")

        if self.sigma_proc is not None:
            self.sigma_proc = np.asarray(self.sigma_proc, dtype=float)
            if self.sigma_proc.shape[0] != S:
                raise ValueError("sigma_proc must have shape (S,).")
            if np.any(self.sigma_proc < 0):
                raise ValueError("All sigma_proc (variances) must be >= 0.")

        if self.kappa is not None:
            self.kappa = np.asarray(self.kappa, dtype=float)
            if self.kappa.shape[0] != S:
                raise ValueError("kappa must have shape (S,).")
            if np.any(self.kappa < 0):
                raise ValueError("All kappa must be >= 0.")

        if self.z_eq is not None:
            self.z_eq = np.asarray(self.z_eq, dtype=float)
            if self.z_eq.shape[0] != S:
                raise ValueError("z_eq must have shape (S,).")

    @property
    def num_states(self) -> int:
        return len(self.states)

    @property
    def mu(self) -> np.ndarray:
        """
        Drift μ_s from canonical ratios r_center,s.
        """
        return np.log(self.r_center) / self.dt_ref

    def ratio_likelihood(self, r_smooth: np.ndarray) -> np.ndarray:
        """
        Phase likelihood matrix L[n,s] from smoothed ratios r_smooth.
        """
        r_smooth = np.asarray(r_smooth, dtype=float)
        N = r_smooth.shape[0]
        S = self.num_states

        L = np.empty((N, S), dtype=float)
        for s in range(S):
            diff = r_smooth - self.r_center[s]
            L[:, s] = np.exp(-0.5 * (diff ** 2) / (self.sigma_r[s] ** 2))

        return L

    def classify_phase_from_likelihoods(
        self,
        L: np.ndarray,
        mode: str = "hard"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simple one-layer classifier from likelihoods L[n,s]:

            p_{n,s} = L[n,s] / sum_s L[n,s]

        mode="hard": S_hat[n] = argmax_s p_{n,s}.
        Returns both S_hat and full probs.
        """
        L = np.asarray(L, dtype=float)
        if L.ndim != 2 or L.shape[1] != self.num_states:
            raise ValueError("L must have shape (N, S).")

        row_sum = L.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0.0] = 1e-12
        probs = L / row_sum
        S_hat = probs.argmax(axis=1)

        if mode not in ("hard", "soft"):
            raise ValueError("mode must be 'hard' or 'soft'.")

        return S_hat, probs

    def sample_phase_sequence(
        self,
        N: int,
        pi0: Optional[np.ndarray] = None,
        random_state: Optional[np.random.Generator | int] = None
    ) -> np.ndarray:
        """
        Sample S_0,...,S_N from Markov chain, mainly for simulation.
        """
        rng = get_rng(random_state)
        S = self.num_states

        if pi0 is None:
            eigvals, eigvecs = np.linalg.eig(self.P.T)
            idx = np.argmin(np.abs(eigvals - 1.0))
            v = np.real(eigvecs[:, idx])
            v = np.clip(v, 0, None)
            if v.sum() == 0:
                pi0 = np.full(S, 1.0 / S)
            else:
                pi0 = v / v.sum()
        else:
            pi0 = np.asarray(pi0, dtype=float)
            if pi0.shape[0] != S:
                raise ValueError("pi0 must have shape (S,).")
            if np.any(pi0 < 0):
                raise ValueError("pi0 entries must be non-negative.")
            if pi0.sum() == 0:
                raise ValueError("pi0 must not be all zeros.")
            pi0 = pi0 / pi0.sum()

        S_idx = np.empty(N + 1, dtype=int)
        S_idx[0] = rng.choice(S, p=pi0)

        for n in range(N):
            i = S_idx[n]
            S_idx[n + 1] = rng.choice(S, p=self.P[i])

        return S_idx


# ---------------------------------------------------------------------------
# 3) Single-axis latent dynamics and observation model
# ---------------------------------------------------------------------------

@dataclass
class SingleAxisLatentModel:
    """
    Latent log-normalized process z_n with phase-dependent drift and noise.
    """
    phase_model: PBPhaseModel

    def simulate(
        self,
        t: np.ndarray,
        S_idx: np.ndarray,
        z0: float = 0.0,
        use_mean_reversion: bool = False,
        random_state: Optional[np.random.Generator | int] = None
    ) -> np.ndarray:
        """
        Simulate z_n over given times t_n and phase sequence S_n.
        """
        rng = get_rng(random_state)
        t = np.asarray(t, dtype=float)
        S_idx = np.asarray(S_idx, dtype=int)

        if t.shape[0] != S_idx.shape[0]:
            raise ValueError("t and S_idx must have the same length (N+1).")
        assert np.all(np.diff(t) > 0), "t must be strictly increasing."

        N = t.shape[0] - 1
        z = np.empty(N + 1, dtype=float)
        z[0] = z0

        mu_s = self.phase_model.mu
        sigma_proc = self.phase_model.sigma_proc  # variances σ_s^2
        if sigma_proc is None:
            raise ValueError("phase_model.sigma_proc must be specified.")

        if use_mean_reversion:
            if self.phase_model.kappa is None or self.phase_model.z_eq is None:
                raise ValueError("kappa and z_eq must be provided for mean-reversion.")

        for n in range(N):
            dt = t[n + 1] - t[n]
            s = S_idx[n]
            drift = mu_s[s] * dt

            if use_mean_reversion:
                kappa_s = self.phase_model.kappa[s]
                z_eq_s = self.phase_model.z_eq[s]
                drift -= kappa_s * (z[n] - z_eq_s) * dt

            std_proc = np.sqrt(sigma_proc[s] * dt)
            eta_n = rng.normal(loc=0.0, scale=std_proc)

            z[n + 1] = z[n] + drift + eta_n

        return z


@dataclass
class ObservationModel:
    """
    Log observation model:

        y_log_n = c + z_n + ε_n,   ε_n ~ N(0, σ_meas^2)
        y_n     = exp(y_log_n)

    with prior on baseline:

        c ~ N(μ_c, σ_c^2)
    """
    sigma_meas: float
    mu_c: float
    sigma_c: float

    def sample_c(
        self,
        random_state: Optional[np.random.Generator | int] = None
    ) -> float:
        rng = get_rng(random_state)
        return float(rng.normal(loc=self.mu_c, scale=self.sigma_c))

    def sample_observations(
        self,
        z: np.ndarray,
        c: float,
        random_state: Optional[np.random.Generator | int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        rng = get_rng(random_state)
        z = np.asarray(z, dtype=float)
        N1 = z.shape[0]

        eps = rng.normal(loc=0.0, scale=self.sigma_meas, size=N1)
        y_log = c + z + eps
        y = np.exp(y_log)
        return y, y_log


# ---------------------------------------------------------------------------
# 4) Multi-axis PB Prevention Core
# ---------------------------------------------------------------------------

@dataclass
class PBPreventionCore:
    """
    PB Prevention Core with 4D latent vector Z^(PB)_n = [I,M,C,B]^T.
    """
    y_PB: np.ndarray
    y_base_PB: np.ndarray

    alpha_I: float = 1.0
    alpha_M: float = 1.0
    alpha_C: float = 1.0
    beta_B: float = 1.0

    delta_Z_min: Optional[np.ndarray] = None
    delta_Z_max: Optional[np.ndarray] = None

    enforce_condition_37: bool = False

    def __post_init__(self):
        self.y_PB = np.asarray(self.y_PB, dtype=float)
        self.y_base_PB = np.asarray(self.y_base_PB, dtype=float)

        if self.y_PB.ndim != 2 or self.y_PB.shape[1] != 4:
            raise ValueError("y_PB must have shape (N+1, 4) for axes [I,M,C,B].")
        if self.y_base_PB.shape != (4,):
            raise ValueError("y_base_PB must have shape (4,).")
        if np.any(self.y_PB <= 0) or np.any(self.y_base_PB <= 0):
            raise ValueError("All y^(PB) and baselines must be > 0.")

        if self.delta_Z_min is not None:
            self.delta_Z_min = np.asarray(self.delta_Z_min, dtype=float)
            if self.delta_Z_min.shape != (4,):
                raise ValueError("delta_Z_min must have shape (4,).")

        if self.delta_Z_max is not None:
            self.delta_Z_max = np.asarray(self.delta_Z_max, dtype=float)
            if self.delta_Z_max.shape != (4,):
                raise ValueError("delta_Z_max must have shape (4,).")

    def compute_Z_PB(self) -> np.ndarray:
        """
        Log-normalized PB latent vector Z^(PB)_n.
        """
        ratio = self.y_PB / self.y_base_PB[None, :]
        Z_PB = np.log(ratio)
        return Z_PB

    def compute_potential(self, Z_PB: np.ndarray) -> np.ndarray:
        """
        Potential V_n = α_I I + α_M M + α_C C - β_B B.
        """
        I = Z_PB[:, 0]
        M = Z_PB[:, 1]
        C = Z_PB[:, 2]
        B = Z_PB[:, 3]
        V = (
            self.alpha_I * I
            + self.alpha_M * M
            + self.alpha_C * C
            - self.beta_B * B
        )
        return V

    @staticmethod
    def compute_gradient_V(
        V: np.ndarray,
        t: np.ndarray
    ) -> np.ndarray:
        """
        Discrete-time gradient ∇V_n.
        """
        V = np.asarray(V, dtype=float)
        t = np.asarray(t, dtype=float)
        if V.shape != t.shape:
            raise ValueError("V and t must have the same shape (N+1,).")
        assert np.all(np.diff(t) > 0), "t must be strictly increasing."

        N = V.shape[0] - 1
        grad = np.zeros_like(V)
        dt = np.diff(t)
        grad[1:] = (V[1:] - V[:-1]) / dt
        return grad

    def compute_delta_Z_single_axis_I(
        self,
        grad_V_n: float,
        V_n: float,
        V_prev: float,
        k: float
    ) -> np.ndarray:
        """
        Single-axis intervention on I_n, with optional enforcement of condition (37).
        """
        delta_I = -k * grad_V_n
        delta_M = 0.0
        delta_C = 0.0
        delta_B = 0.0

        if self.enforce_condition_37:
            delta_V = V_n - V_prev
            if delta_V > 0 and self.alpha_I > 0:
                limit_I = -delta_V / self.alpha_I
                if delta_I > limit_I:
                    delta_I = limit_I

        delta_Z = np.array([delta_I, delta_M, delta_C, delta_B], dtype=float)

        if self.delta_Z_min is not None:
            delta_Z = np.maximum(delta_Z, self.delta_Z_min)
        if self.delta_Z_max is not None:
            delta_Z = np.minimum(delta_Z, self.delta_Z_max)

        return delta_Z

    def apply_intervention_over_time(
        self,
        t: np.ndarray,
        k: float,
        mode: str = "single_axis_I"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply PB prevention over time, returning updated Z_PB, V, ΔZ, and ΔP.
        """
        t = np.asarray(t, dtype=float)
        assert np.all(np.diff(t) > 0), "t must be strictly increasing."

        Z_PB = self.compute_Z_PB()
        V = self.compute_potential(Z_PB)
        grad_V = self.compute_gradient_V(V, t)

        N1 = Z_PB.shape[0]
        delta_Z_all = np.zeros_like(Z_PB)
        delta_P = np.zeros(N1, dtype=float)

        for n in range(1, N1):
            if grad_V[n] <= 0:
                continue

            delta_P[n] = k * grad_V[n]

            if mode == "single_axis_I":
                delta_Z = self.compute_delta_Z_single_axis_I(
                    grad_V_n=grad_V[n],
                    V_n=V[n],
                    V_prev=V[n - 1],
                    k=k
                )
            else:
                raise NotImplementedError(f"Mode '{mode}' not implemented.")

            Z_PB[n] = Z_PB[n] + delta_Z
            V[n] = self.compute_potential(Z_PB[n:n+1])[0]
            delta_Z_all[n] = delta_Z

        return Z_PB, V, delta_Z_all, delta_P


# ---------------------------------------------------------------------------
# CLI utilities and runner
# ---------------------------------------------------------------------------

def parse_array(arg: str, dtype=float) -> np.ndarray:
    """
    Parse a comma-separated string into a numpy array.
    Example: "0,1,2.5" -> array([0.0, 1.0, 2.5])
    """
    parts = [p.strip() for p in arg.split(",") if p.strip() != ""]
    return np.array(parts, dtype=dtype)


def run_from_cli(args):
    """
    Main pipeline, driven by command-line arguments.
    Saves CSVs and plots into args.outdir.
    """
    rng = get_rng(42)

    # -------- Input parsing --------
    t = parse_array(args.times, dtype=float)
    y = parse_array(args.values, dtype=float)
    if t.shape != y.shape:
        raise ValueError("times and values must have the same length.")
    assert np.all(np.diff(t) > 0), "t must be strictly increasing."

    healthy_indices = parse_array(args.healthy, dtype=int)

    r_band_low = args.r_band_low
    r_band_high = args.r_band_high
    W_smooth = args.window
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # -------- ObservedSeries layer --------
    obs = ObservedSeries(y=y, t=t, healthy_indices=healthy_indices, W=W_smooth)
    obs.compute_baseline()
    c_est = np.log(obs.y_base)
    _, z_obs = obs.compute_normalized_log(c=c_est)
    r, ell = obs.compute_ratios()
    ell_smooth = obs.compute_smoothed_log_ratios()
    EI = obs.compute_EI(r_band_low=r_band_low, r_band_high=r_band_high)
    T_exit = obs.exit_time_from_health_basin(theta_H=args.theta_H)

    # -------- PB phase model and classification --------
    states = ["Health", "EarlyDrift"]
    P = np.array([[0.95, 0.05],
                  [0.10, 0.90]])
    r_min = np.array([0.95, 1.20])
    r_max = np.array([1.05, 1.60])
    r_center = np.array([1.00, 1.35])
    sigma_r = np.array([0.05, 0.10])
    sigma_proc = np.array([0.001, 0.01])  # variances

    phase_model = PBPhaseModel(
        states=states,
        P=P,
        r_min=r_min,
        r_max=r_max,
        r_center=r_center,
        sigma_r=sigma_r,
        dt_ref=1.0,
        sigma_proc=sigma_proc,
    )

    # r_smooth ≈ exp(ℓ̃_n)
    r_smooth = np.exp(ell_smooth)
    L = phase_model.ratio_likelihood(r_smooth=r_smooth)
    S_hat, phase_probs = phase_model.classify_phase_from_likelihoods(L, mode="hard")

    # -------- Latent single-axis simulation (demo) --------
    latent_model = SingleAxisLatentModel(phase_model=phase_model)
    # S_hat has length N = len(y)-1; we want S_idx_for_sim of length N+1
    # with S_idx_for_sim[n] ~ S_hat[n] for n=0..N-1:
    S_idx_for_sim = np.concatenate([S_hat, [S_hat[-1]]])
    z_sim = latent_model.simulate(
        t=t,
        S_idx=S_idx_for_sim,
        z0=0.0,
        use_mean_reversion=False,
        random_state=rng
    )

    # -------- Observation model for synthetic noisy y (demo) --------
    obs_model = ObservationModel(
        sigma_meas=args.sigma_meas,
        mu_c=args.mu_c,
        sigma_c=args.sigma_c,
    )
    c_sampled = obs_model.sample_c(random_state=rng)
    y_sim, y_log_sim = obs_model.sample_observations(z=z_sim, c=c_sampled, random_state=rng)

    # -------- PB Prevention Core (4D) --------
    if args.use_prevention_core:
        # Build simple 4D PB axes from y
        y_I = y
        y_M = 0.8 * y + 0.2
        y_C = 1.2 * y + 0.1
        y_B = 1.0 / (0.5 + 0.5 * y)
        y_PB = np.stack([y_I, y_M, y_C, y_B], axis=1)

        y_base_PB = y_PB[healthy_indices].mean(axis=0)

        pb_core = PBPreventionCore(
            y_PB=y_PB,
            y_base_PB=y_base_PB,
            alpha_I=1.0,
            alpha_M=1.0,
            alpha_C=1.0,
            beta_B=1.0,
            delta_Z_min=np.array([-0.5, -0.2, -0.2, -0.2]),
            delta_Z_max=np.array([0.0, 0.0, 0.0, 0.0]),
            enforce_condition_37=True,
        )

        Z_PB_before = pb_core.compute_Z_PB()
        V_before = pb_core.compute_potential(Z_PB_before)
        grad_V_before = pb_core.compute_gradient_V(V_before, t)

        Z_PB_after, V_after, delta_Z_all, delta_P = pb_core.apply_intervention_over_time(
            t=t,
            k=args.k_gain,
            mode="single_axis_I",
        )
    else:
        Z_PB_before = V_before = grad_V_before = Z_PB_after = V_after = delta_Z_all = delta_P = None

    # ------------------------------------------------------------------
    # Save CSV outputs
    # ------------------------------------------------------------------
    N = len(t) - 1
    # pad step-based quantities to length N+1
    r_full = np.concatenate([r, [np.nan]])
    ell_full = np.concatenate([ell, [np.nan]])
    ell_smooth_full = np.concatenate([ell_smooth, [np.nan]])
    EI_full = np.concatenate([EI, [np.nan]])
    # ✅ corrected: phase_full aligned with steps [t_i, t_{i+1}]
    phase_full = np.concatenate([S_hat, [S_hat[-1]]])

    main_data = np.column_stack([
        t,
        y,
        z_obs,
        r_full,
        ell_full,
        ell_smooth_full,
        EI_full,
        phase_full,
        z_sim,
        y_sim
    ])
    header = "t,y,z_obs,r,ell,ell_smooth,EI,phase_idx,z_sim,y_sim"
    np.savetxt(
        os.path.join(outdir, "timeseries.csv"),
        main_data,
        delimiter=",",
        header=header,
        comments=""
    )

    if args.use_prevention_core:
        pb_data = np.column_stack([
            t,
            V_before,
            V_after,
            grad_V_before,
            delta_P,
            delta_Z_all[:, 0],
            delta_Z_all[:, 1],
            delta_Z_all[:, 2],
            delta_Z_all[:, 3],
        ])
        pb_header = "t,V_before,V_after,grad_V_before,delta_P,delta_Z_I,delta_Z_M,delta_Z_C,delta_Z_B"
        np.savetxt(
            os.path.join(outdir, "pb_core.csv"),
            pb_data,
            delimiter=",",
            header=pb_header,
            comments=""
        )

    # ------------------------------------------------------------------
    # Save plots
    # ------------------------------------------------------------------

    # Observed y
    plt.figure()
    plt.plot(t, y, marker="o")
    plt.xlabel("time")
    plt.ylabel("y (observed)")
    plt.title("Observed biomarker")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_y.png"))
    plt.close()

    # Latent z (observed vs simulated)
    plt.figure()
    plt.plot(t, z_obs, marker="o", label="z_obs")
    plt.plot(t, z_sim, marker="x", label="z_sim (simulated)")
    plt.xlabel("time")
    plt.ylabel("z")
    plt.title("Latent log-normalized process")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_z.png"))
    plt.close()

    # Phase index
    plt.figure()
    plt.step(t, phase_full, where="post")
    plt.xlabel("time")
    plt.ylabel("phase index")
    plt.title("Estimated phase (simple classifier)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "plot_phase.png"))
    plt.close()

    # PB potential and gradient (if enabled)
    if args.use_prevention_core:
        plt.figure()
        plt.plot(t, V_before, marker="o", label="V_before")
        plt.plot(t, V_after, marker="x", label="V_after")
        plt.xlabel("time")
        plt.ylabel("V")
        plt.title("PB Potential before/after intervention")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "plot_V.png"))
        plt.close()

        plt.figure()
        plt.plot(t, grad_V_before, marker="o")
        plt.xlabel("time")
        plt.ylabel("grad V")
        plt.title("Gradient of PB Potential")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "plot_gradV.png"))
        plt.close()

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("=== PB Dynamic Model v5 ===")
    print(f"Number of time points: {len(t)}")
    print(f"Baseline y_base (from H): {obs.y_base:.3f}")
    print(f"Estimated c_est = log(y_base): {c_est:.3f}")
    print(f"Exit time from health basin (theta_H={args.theta_H}): {T_exit}")
    print(f"Outputs saved in folder: {outdir}")


def main():
    parser = argparse.ArgumentParser(description="PB Dynamic Model v5 - CLI runner")

    parser.add_argument(
        "--times",
        type=str,
        required=True,
        help="Comma-separated time points, e.g. '0,1,2,3'"
    )
    parser.add_argument(
        "--values",
        type=str,
        required=True,
        help="Comma-separated biomarker values (same length as times), e.g. '1.0,0.9,1.1,1.3'"
    )
    parser.add_argument(
        "--healthy",
        type=str,
        default="0,1,2",
        help="Comma-separated indices for healthy baseline window, e.g. '0,1,2'"
    )

    parser.add_argument("--r_band_low", type=float, default=1.27,
                        help="Lower bound for early-drift band (default 1.27)")
    parser.add_argument("--r_band_high", type=float, default=1.40,
                        help="Upper bound for early-drift band (default 1.40)")
    parser.add_argument("--window", type=int, default=3,
                        help="Window size for smoothing / EI (default 3)")
    parser.add_argument("--theta_H", type=float, default=0.5,
                        help="Health basin threshold for |z| (default 0.5)")

    parser.add_argument("--sigma_meas", type=float, default=0.05,
                        help="Measurement noise std in log-space (default 0.05)")
    parser.add_argument("--mu_c", type=float, default=0.0,
                        help="Prior mean for baseline c (default 0.0)")
    parser.add_argument("--sigma_c", type=float, default=0.2,
                        help="Prior std for baseline c (default 0.2)")

    parser.add_argument("--use_prevention_core", action="store_true",
                        help="If set, run the 4D PB Prevention Core")
    parser.add_argument("--k_gain", type=float, default=0.5,
                        help="Intervention gain k for PB Prevention Core (default 0.5)")

    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs",
        help="Output directory for CSVs and plots (will be created if needed)"
    )

    args = parser.parse_args()
    run_from_cli(args)


if __name__ == "__main__":
    main()
