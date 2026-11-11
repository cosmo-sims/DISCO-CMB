"""The grids of a CMB computation, shaped by a fiducial cosmology.

The sources have structure at recombination and on the sound-horizon scale in
``tau``, and at the acoustic scale and the Silk damping scale in ``k``; the
grids here are dense where that structure is (:func:`cmb_tau_grid`,
:func:`cmb_k_grid`). The characteristic scales come from one trajectory of
the source kernel itself (:func:`_kernel_thermal_history`), so building them
compiles nothing new. :func:`cmb_grids` bundles them into :class:`CMBGrids`,
built once and shared by every batch of an inference loop.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import discoeb as eb
import numpy as np
from discoeb import TAU_START
from scipy import integrate

from .sources import solve_sources, visibility_on_grid

SUMMARY_GRID = 2048
"""Points of the uniform ``log tau`` grid the grid summary reads the opacity on."""

DEFAULT_CMB_ELLS = np.unique(
    np.concatenate(
        [np.arange(2, 40, 2), np.arange(40, 200, 5), np.arange(200, 2501, 50)]
    )
).astype(np.int64)
"""Default multipoles at which ``C_l`` is evaluated."""


Z_REION_SEARCH_MAX = 30.0
"""Redshift below which a reionization visibility bump would be located."""


TAU_END_MARGIN = 0.01
"""How far past the fiducial cosmology's ``tau0`` the save grid of :func:`cmb_grids`
extends, as a fraction, so that the grids built once from a fiducial serve every
batch whose conformal ages fall within it. Measured against per-batch grids on
the 128 nearby cosmologies of the tests, the shift onto the fiducial's grid
moves ``D_l`` by at most 2.7e-4 at 1 percent, 1.0e-3 at 2 percent and 4.4e-3
at 5 percent (the low multipoles' late-time sources sit on the coarse end of
the grid)."""


def _grhoa4(a, grhog, grhor, grhoc, grhob, grhov):
    """``8 pi G rho a^4``: the expansion rate is ``a' = sqrt(grhoa4 / 3)``."""

    return grhog + grhor + (grhoc + grhob) * a + grhov * a**4


def _dtau_da(a, *dens):
    """``dtau/da = 1 / a'``."""

    return np.sqrt(3.0 / _grhoa4(a, *dens))


SUMMARY_K = 1.0e-3
"""Wave mode (Mpc^-1) of the one trajectory whose recombination block the grid summary reads."""


@functools.lru_cache(maxsize=64)
def _kernel_thermal_history(cosmology, n_grid: int = SUMMARY_GRID, n_save: int = 1000):
    """``(tau_grid, a, opacity, tau0)`` of one cosmology on ``n_grid`` uniform ``log tau`` points, from the sources' own solve.

    One trajectory of :func:`~discocmb.sources.solve_sources` (the kernel the spectra use, so
    nothing more is compiled) records the opacity ``kappa'``, with
    reionization, from its recombination block's dense output; the scale
    factor is the background quadrature's. The kernel's save grid has
    ``n_save`` times, part of its identity, and every launch records the
    start, ``TAU_START``, so the grid is read in launches of ``n_save - 1``
    points.
    """

    a_bg, tau_bg = eb.background_history(cosmology)
    tau0 = float(tau_bg[-1])
    tau_grid = np.exp(np.linspace(np.log(TAU_START), np.log(tau0), n_grid))
    opacity = np.empty(n_grid)
    for c0 in range(0, n_grid, n_save - 1):
        chunk = tau_grid[c0 : c0 + n_save - 1]
        times = np.full(n_save, tau_grid[-1])
        times[0] = TAU_START
        times[1 : 1 + len(chunk)] = chunk
        # equal times past the chunk are not strictly increasing: nudge them
        times[1 + len(chunk) :] += np.arange(1, n_save - len(chunk)) * 1.0e-6
        solution = solve_sources((cosmology,), np.asarray([SUMMARY_K]), times)
        channels = np.asarray(solution.values[0, :, 0, :])  # (6, n_save)
        opacity[c0 : c0 + len(chunk)] = channels[4, 1 : 1 + len(chunk)]
    return tau_grid, np.interp(tau_grid, tau_bg, a_bg), opacity, tau0


def _cmb_grid_summary(cosmology) -> dict[str, float]:
    return _grid_summary(eb.as_cosmology(cosmology))


@functools.lru_cache(maxsize=4096)
def _grid_summary(cosmology) -> dict[str, float]:
    """Return the characteristic scales that set the adaptive ``k`` / ``tau`` grids.

    ``tau_star`` is the visibility peak and ``delta_tau_rec`` its Gaussian
    equivalent width; ``r_s`` is the sound horizon at recombination (which sets
    the acoustic oscillation period in ``k``) and ``k_D`` the Silk damping
    wavenumber (which sets their envelope). Together they say where the CMB
    sources have structure, and hence where the grids must be dense.
    """

    dens = eb.density_coefficients(cosmology)
    tau_grid, a_vals, opacity, tau0 = _kernel_thermal_history(cosmology)

    _, vis, _ = visibility_on_grid(tau_grid, opacity, tau_grid, float(tau0))
    peak = int(np.argmax(vis))
    tau_star = float(tau_grid[peak])
    a_star = float(a_vals[peak])

    half = 0.5 * float(vis[peak])
    left = float(np.interp(half, vis[: peak + 1], tau_grid[: peak + 1]))
    right = float(np.interp(half, vis[peak:][::-1], tau_grid[peak:][::-1]))
    delta_tau_rec = max((right - left) / (2.0 * np.sqrt(2.0 * np.log(2.0))), 1.0)

    grhog_v, grhor_v, grhoc_v, grhob_v, _ = dens
    tau_eq = float(
        np.interp((grhog_v + grhor_v) / (grhoc_v + grhob_v), a_vals, tau_grid)
    )

    # Comoving sound horizon r_s = int c_s dtau = int_0^{a_star} da / (a' sqrt(3(1+R))),
    # with a' = sqrt(grhoa4/3) and R = 3 rho_b / (4 rho_g).
    def sound_integrand(ap):
        r = 0.75 * grhob_v * ap / grhog_v
        return 1.0 / np.sqrt(_grhoa4(ap, *dens) * (1.0 + r))

    r_s = float(integrate.quad(sound_integrand, 0.0, a_star, limit=200)[0])

    # Silk damping scale from the photon diffusion length.
    a_silk = np.linspace(max(float(a_vals[0]), 1.0e-8), a_star, 5000)
    opacity_silk = np.maximum(np.interp(a_silk, a_vals, opacity), 1.0e-30)
    r = 0.75 * grhob_v * a_silk / grhog_v
    integrand_d = (
        (r**2 + 16.0 * (1.0 + r) / 15.0)
        / (6.0 * (1.0 + r) ** 2 * opacity_silk)
        * _dtau_da(a_silk, *dens)
    )
    k_D = float(1.0 / np.sqrt(np.trapezoid(integrand_d, a_silk)))

    # The secondary visibility bump from reionization when tau_reio > 0; without
    # one, the save grid's late-time weight falls back to a broad bump.
    post_rec = a_vals > 1.0 / (1.0 + Z_REION_SEARCH_MAX)
    if np.any(post_rec) and np.max(np.where(post_rec, vis, 0.0)) > 1.0e-3 * vis[peak]:
        reion_idx = int(np.argmax(np.where(post_rec, vis, 0.0)))
        tau_reion = float(tau_grid[reion_idx])
        delta_tau_reion = max(0.03 * tau0, 80.0)
    else:
        tau_reion = 0.6 * tau0
        delta_tau_reion = 0.05 * tau0

    out = {
        "tau0": float(tau0),
        "tau_eq": tau_eq,
        "tau_star": tau_star,
        "delta_tau_rec": delta_tau_rec,
        "tau_reion": tau_reion,
        "delta_tau_reion": delta_tau_reion,
        "r_s": r_s,
        "k_D": k_D,
    }
    return out


def _resample_from_density(x, density, n):
    """Return ``n`` points drawn from ``x`` in proportion to ``density``."""

    cdf = np.cumsum(density) * (x[1] - x[0])
    cdf -= cdf[0]
    cdf /= cdf[-1]
    return np.interp(np.linspace(0.0, 1.0, n), cdf, x)


def cmb_k_grid(
    cosmology,
    *,
    n: int = 200,
    mode: str = "ode",
    k_min: float = 1.0e-5,
    k_max: float = 0.5,
    ell_min: int = 2,
    ell_max: int = 2500,
    n_ell_samples: int = 30,
    n_eval: int = 5000,
) -> np.ndarray:
    """Return a nonuniform wavenumber grid concentrated where the sources vary.

    ``mode="ode"`` resolves the acoustic oscillations of the source itself
    (period set by ``r_s``, envelope by Silk damping ``k_D``) and is the grid the
    perturbation solve runs on. ``mode="cl"`` additionally weights each ``k`` by
    how many multipoles project onto it, ``k ~ l / chi_star``, and is the finer
    grid the ``C_l`` quadrature uses after the sources are interpolated.
    """

    cosmology = eb.as_cosmology(cosmology)
    summary = _cmb_grid_summary(cosmology)
    x = np.linspace(np.log(k_min), np.log(k_max), n_eval)
    k = np.exp(x)
    primordial = k ** (cosmology.n_s + 2.0)
    acoustic_curv = (1.0 / summary["r_s"]) ** 2
    damped = primordial * np.exp(-((k / summary["k_D"]) ** 2))

    if mode == "cl":
        sigma_k = 1.0 / summary["delta_tau_rec"]
        chi_star = summary["tau0"] - summary["tau_star"]
        ell_samples = np.unique(
            np.geomspace(ell_min, ell_max, n_ell_samples).astype(int)
        )
        raw_weight = np.zeros_like(k)
        for ell in ell_samples:
            envelope = np.exp(-0.5 * ((k - ell / chi_star) / (3.0 * sigma_k)) ** 2)
            raw_weight += np.maximum(acoustic_curv, sigma_k**2) * envelope * damped
        floor = 1.0e-6 * np.max(raw_weight)
    else:
        smooth_curv = (k * summary["r_s"]) ** 2 / summary["tau_eq"] ** 2
        raw_weight = np.maximum(acoustic_curv, smooth_curv) * damped
        floor = 0.005 * np.max(raw_weight)

    grid = np.exp(_resample_from_density(x, (raw_weight + floor) ** (1.0 / 3.0), n))
    grid[0], grid[-1] = k_min, k_max
    return grid


def cmb_tau_grid(
    cosmology, k_values, n: int, *, tau_end: float | None = None, n_eval: int = 10000
) -> np.ndarray:
    """Return a nonuniform conformal-time grid on which to save the sources.

    The sources are sharply peaked at recombination (width ``delta_tau_rec``),
    modulated on the sound-horizon scale, and would have a secondary bump at
    reionization, so the sampling density is the cube root of the sum of those
    three weights. Endpoints are pinned to ``[TAU_START, tau_end]``, which the
    solver's save grid requires.

    ``tau_end`` defaults to this cosmology's ``tau0``. A batch of cosmologies
    passes the *largest* ``tau0`` of the batch so that no trajectory's save grid
    is cut short; the shaping weights still come from ``cosmology``.
    """

    k_arr = np.asarray(k_values, dtype=np.float64)
    s = _cmb_grid_summary(cosmology)
    tau0 = s["tau0"] if tau_end is None else float(tau_end)

    tau = np.linspace(TAU_START, tau0, n_eval, dtype=np.float64)
    weight = (
        np.exp(-0.5 * ((tau - s["tau_star"]) / s["delta_tau_rec"]) ** 2)
        / s["delta_tau_rec"] ** 2
    )
    weight += (
        np.exp(-0.5 * ((tau - s["tau_star"]) / s["r_s"]) ** 2)
        * (float(k_arr[-1]) / np.sqrt(3.0)) ** 2
    )
    weight += (
        0.3
        * np.exp(-0.5 * ((tau - s["tau_reion"]) / s["delta_tau_reion"]) ** 2)
        / s["delta_tau_reion"] ** 2
    )

    tau_save = _resample_from_density(
        tau, (weight + 0.005 * np.max(weight)) ** (1.0 / 3.0), n
    )
    tau_save[0], tau_save[-1] = TAU_START, tau0
    return tau_save


class CMBGrids(NamedTuple):
    """The grids of one CMB spectrum computation, shaped by a fiducial cosmology.

    ``k_values`` are the wave modes the hierarchy is solved on, ``k_fine`` the
    quadrature grid in ``k`` the sources are refined onto, ``tau_save`` the
    conformal times the sources are recorded at, ``ells`` the multipoles.
    Built by :func:`cmb_grids`; passed to :func:`~discocmb.main.compute_cl_power_spectrum_batch`
    they fix the line-of-sight projection operator across batches.
    """

    k_values: np.ndarray
    ells: np.ndarray
    k_fine: np.ndarray
    tau_save: np.ndarray

    @property
    def tau_end(self) -> float:
        return float(self.tau_save[-1])


def cmb_grids(
    cosmology,
    *,
    k_values: np.ndarray | None = None,
    ells: np.ndarray | None = None,
    n_save: int = 1000,
    n_k_fine: int = 1500,
    k_fine_values: np.ndarray | None = None,
    tau_end: float | None = None,
    tau_end_margin: float = TAU_END_MARGIN,
) -> CMBGrids:
    """Return the :class:`CMBGrids` shaped by ``cosmology``, built once for many batches.

    The grids adapt to where the sources have structure -- the recombination
    epoch and the sound horizon in ``tau`` (:func:`cmb_tau_grid`), the acoustic
    scale and Silk damping in ``k`` (:func:`cmb_k_grid`) -- and nearby
    cosmologies share that structure, so one fiducial's grids serve them all;
    a batch has always been solved on its first cosmology's grids. Fixing them
    across batches is what lets the projection operator
    (:func:`~discocmb.transfer.projection_operator`: the Bessel table and the ``ln k`` refinement
    matrix) be built once rather than per batch, 0.3 s each time.

    The save grid ends at ``tau_end``, by default ``tau_end_margin`` past the
    fiducial's ``tau0``, and every cosmology projected on these grids must
    have ``tau0 <= tau_end`` (its sources are shifted onto the shared
    comoving-distance grid, which must reach ``tau0``). A cosmology's own
    trajectory still stops at its own ``tau0``; the saves past it carry zero
    sources. The shift grows with the margin and costs accuracy through the
    source interpolation (see :data:`TAU_END_MARGIN` for the measured cost),
    so keep the margin to what the batches need. A cosmology past ``tau_end``
    raises ``ValueError`` rather than projecting short.
    """

    cosmology = eb.as_cosmology(cosmology)
    if k_values is None:
        k_values = cmb_k_grid(cosmology, n=128, mode="ode")
    if ells is None:
        ells = DEFAULT_CMB_ELLS
    k_values = np.asarray(k_values, dtype=np.float64)
    ells = np.asarray(ells, dtype=np.int64)
    if tau_end is None:
        tau_end = _cmb_grid_summary(cosmology)["tau0"] * (1.0 + float(tau_end_margin))
    tau_save = cmb_tau_grid(cosmology, k_values, n_save, tau_end=float(tau_end))
    if k_fine_values is None:
        k_fine = cmb_k_grid(
            cosmology,
            n=n_k_fine,
            mode="cl",
            k_min=float(k_values[0]),
            k_max=float(k_values[-1]),
            ell_max=int(np.max(ells)),
        )
    else:
        k_fine = np.asarray(k_fine_values, dtype=np.float64)
    return CMBGrids(k_values, ells, k_fine, tau_save)
