"""Line-of-sight integration: the temperature and E-mode transfer functions.

Rather than following the photon multipole hierarchy up to ``l ~ 2500``, the
temperature transfer function is a single integral of the source functions
(:mod:`discocmb.sources`) against spherical Bessel functions,

    ``Theta_l(k) = int_0^{tau_0} dtau [S j_l(x) + S_j1 j_l'(x) + S_j2 j_l''(x)]``

with ``x = k (tau_0 - tau)``. The E-mode transfer function is the
quadrupole/polarization term against ``j_l(x) / x^2``,

    ``E_l(k) = sqrt((l+2)!/(l-2)!) int_0^{tau_0} dtau S_j2 j_l(x) / x^2``

(Zaldarriaga & Seljak 1997; CAMB's ``sources(2)``), so the projection carries
a fourth Bessel row, ``j_l(x) / x^2`` with the ``l``-dependent prefactor folded
in, tabulated directly (not divided after interpolation, since ``j_l`` is
interpolated linearly and ``j_2(x) / x^2 -> 1/15`` at the end of the line of
sight, where a reionized universe still has visibility).

**How the line of sight is projected.** As matrix products on grids shared by
the whole batch (:func:`projection_operator`, :func:`theta_ell_batch`). The
sources of every cosmology are shifted in ``tau`` onto one comoving-distance
grid ``chi = tau0_ref - tau`` (a 4-point Lagrange stencil; each cosmology's own
``tau0`` differs from the reference by up to ~100 Mpc), spline-interpolated in
``ln k`` from the solved modes to the fine quadrature grid by a precomputed
matrix, and contracted against one weighted table of ``j_l``, ``j_l'`` and
``j_l''`` on the ``(k, chi)`` grid -- a batched GEMM over cosmologies. Against a
per-cosmology projection with exact ``chi`` this costs 2e-4 in ``D_l``, and the
default ``float32`` arithmetic another 2.6e-4, both below the pipeline's other
numerical differences; it runs in 0.25 ms per cosmology (2.3 ms in ``float64``)
where the per-cosmology spline and Bessel evaluation took 13 ms. An FFTLog
projection was tried and rejected: it must resample the sources onto a
log-spaced ``chi`` grid that is coarsest at recombination, needs 32768 points to
match, and even then costs an order of magnitude more than the table
contraction, whose flop count is a few percent of the FFT's.
"""

from __future__ import annotations

import functools
import math
import os
from concurrent.futures import ThreadPoolExecutor
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from discoeb.spline_interpolation import spline_interpolation
from scipy import special

from .sources import assemble_sources

CACHE_SIZE = 8
"""Batches of grids whose projection operators and lensing tables are kept."""


SHIFT_STENCIL = 4
"""Points of the Lagrange stencil that shifts a cosmology's sources in ``tau`` onto
the shared ``chi`` grid: 4 (cubic) leaves 2e-4 in ``D_l``, 2 (linear) 7e-3."""

PROJECTION_BUDGET_BYTES = 1.0e9
"""Device memory for one projection chunk's fine-``k`` source arrays, which sets how
many fine wave modes are contracted together at once."""


def array_cache(function):
    """:func:`functools.lru_cache` of :data:`CACHE_SIZE` for a function of NumPy arrays.

    Array arguments are keyed by dtype, shape and contents and handed to
    ``function`` as read-only copies; other arguments must be hashable.
    """

    def key(value):
        if isinstance(value, np.ndarray):
            return (np.ndarray, value.dtype.str, value.shape, value.tobytes())
        return value

    def unkey(value):
        if isinstance(value, tuple) and len(value) == 4 and value[0] is np.ndarray:
            _, dtype, shape, data = value
            return np.frombuffer(data, dtype=dtype).reshape(shape)
        return value

    @functools.lru_cache(maxsize=CACHE_SIZE)
    def cached(*keys):
        return function(*(unkey(k) for k in keys))

    @functools.wraps(function)
    def wrapper(*args):
        return cached(*(key(a) for a in args))

    wrapper.cache_clear = cached.cache_clear
    return wrapper


def trapezoid_weights(x, xp=np):
    """Weights ``w`` of the trapezoid rule on the samples ``x``: ``int f ~ sum_j w_j f(x_j)``.

    ``xp`` is ``np`` on the host or ``jnp`` under ``jit``. The weights are
    positive whichever way ``x`` runs.
    """

    half = 0.5 * xp.abs(xp.diff(x))
    return xp.pad(half, (0, 1)) + xp.pad(half, (1, 0))


def primordial_spectrum(k, n_s, A_s, k_pivot):
    """``P_R(k) = A_s (k / k_pivot)^(n_s - 1)`` of a batch, shape ``(n_c, *k.shape)``.

    ``n_s`` and ``A_s`` are ``(n_c,)``; ``k`` is any array of wave numbers.
    """

    batch = (slice(None),) + (None,) * jnp.ndim(k)
    return A_s[batch] * (k / k_pivot) ** (n_s[batch] - 1.0)


def _bessel_row(ell: int, n_x: int, dx: float) -> np.ndarray:
    """Return ``j_ell`` sampled on the uniform grid ``x = 0, dx, ..., dx (n_x-1)``.

    ``j_ell(x)`` is exponentially small below its turning point ``x ~ ell``, so
    the row is left at zero until ``ell - 4 ell^{1/3}``.
    """

    x = np.linspace(0.0, dx * (n_x - 1), n_x)
    x_safe = np.where(x > 1.0e-30, x, 1.0)
    pref = np.where(x > 1.0e-30, np.sqrt(np.pi / (2.0 * x_safe)), 0.0)
    values = np.zeros_like(x)
    x_min = max(0.0, float(ell) - 4.0 * float(ell) ** (1.0 / 3.0))
    start = max(0, int(x_min / dx) - 1)
    values[start:] = pref[start:] * special.jv(float(ell) + 0.5, x[start:])
    if ell == 0:
        values[0] = 1.0
    return values


@array_cache
def _bessel_tables(ell_arr, x_max, dx):
    """Uniform ``j_ell``, ``j_{ell+1}`` and ``j_ell / x^2`` interpolation tables to ``x_max``."""

    n_x = int(np.ceil(x_max / dx)) + 2
    unique = [int(ell) for ell in np.unique(np.concatenate([ell_arr, ell_arr + 1]))]
    # Each row is a few hundred thousand evaluations of scipy's j_nu, which
    # releases the GIL: the rows are computed on every core (85 s on one).
    # j_l / x^2 is the same row divided through, so it is not recomputed.
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        rows = dict(
            zip(unique, pool.map(lambda ell: _bessel_row(ell, n_x, dx), unique))
        )
    return (
        1.0 / dx,
        n_x,
        jnp.asarray(np.stack([rows[int(ell)] for ell in ell_arr])),
        jnp.asarray(np.stack([rows[int(ell) + 1] for ell in ell_arr])),
        jnp.asarray(
            np.stack([_over_x2(rows[int(ell)], int(ell), n_x, dx) for ell in ell_arr])
        ),
    )


def _over_x2(row: np.ndarray, ell: int, n_x: int, dx: float) -> np.ndarray:
    """``j_ell(x) / x^2`` from the ``j_ell`` row: the E-mode kernel.

    Smooth through ``x = 0`` (``1/15`` there for ``ell = 2``, zero above), so
    it interpolates accurately where ``j_ell`` itself, divided afterwards,
    would not.
    """

    x = np.linspace(0.0, dx * (n_x - 1), n_x)
    x_safe = np.where(x > 1.0e-30, x, 1.0)
    values = row / x_safe**2
    values[0] = 1.0 / 15.0 if ell == 2 else 0.0
    return values


def _uniform_stencil(x, inv_dx, n_x):
    """``(idx, frac)`` that interpolate a table sampled uniformly from ``x = 0`` linearly at ``x``.

    Shared by every table on the same ``x``: ``(1 - frac) vals[idx] + frac vals[idx + 1]``.
    """

    u = x * inv_dx
    idx = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, n_x - 2)
    return idx, jnp.clip(u - idx, 0.0, 1.0)


@jax.jit
def k_band(ells, k, chi_max, chi_star):
    """Return the ``(n_ell, n_k)`` mask of wave modes that project onto each multipole.

    ``j_l(x)`` is exponentially small below its turning point, so modes with
    ``k chi_max < l - 4 l^{1/3}`` contribute nothing; and modes far above
    ``k ~ l / chi_star`` only contribute through rapidly oscillating tails that
    the finite ``tau`` and ``k`` grids cannot resolve, so integrating them
    injects aliasing noise rather than signal. Without the upper cut the damping
    tail is inflated by several percent (``+6%`` at ``l = 500``).

    ``chi_star`` may carry leading batch axes, ``(n_c,)`` for a batch of
    cosmologies, and the mask then has them too: ``(n_c, n_ell, n_k)``, from
    one jitted call rather than one eager call per cosmology.
    """

    ells = ells[:, None]
    chi_star = jnp.asarray(chi_star)[..., None, None]
    k_lo = jnp.maximum(0.0, ells - 4.0 * ells ** (1.0 / 3.0)) / chi_max
    k_hi = (ells + 2500.0) / chi_star
    return (k[None, :] >= k_lo) & (k[None, :] <= k_hi)


def lagrange_shift_stencil(tau, delta, order=SHIFT_STENCIL):
    """Interpolate a ``tau``-sampled function at ``tau + delta``: stencil start indices and weights.

    Returns ``(i0, w)`` with ``i0`` of shape ``(n_tau,)`` and ``w`` of shape
    ``(n_tau, order)`` such that ``f(tau_j + delta) ~ sum_a w[j, a] f(tau[i0_j + a])``.
    Targets outside the grid are clamped to its ends. ``delta`` may carry
    leading axes, ``(n_c,)`` for one shift per cosmology of a batch, which
    then lead the shapes of ``i0`` and ``w``.
    """

    tau = np.asarray(tau, dtype=np.float64)
    n = len(tau)
    target = np.clip(
        np.asarray(delta, dtype=np.float64)[..., None] + tau, tau[0], tau[-1]
    )
    i0 = np.clip(np.searchsorted(tau, target) - order // 2, 0, n - order)
    w = np.ones(target.shape + (order,))
    for a in range(order):
        for b in range(order):
            if a != b:
                w[..., a] *= (target - tau[i0 + b]) / (tau[i0 + a] - tau[i0 + b])
    return i0, w


def shift_onto_chi(x, shift_idx, shift_w):
    """``x`` of a batch, ``(n_c, ..., n_tau)``, shifted onto the shared ``chi`` grid along its last axis.

    ``shift_idx`` ``(n_c, n_tau)`` and ``shift_w`` ``(n_c, n_tau, order)`` are
    each cosmology's stencil from :func:`lagrange_shift_stencil`.
    """

    lead = (slice(None),) + (None,) * (x.ndim - 2)
    return sum(
        shift_w[lead + (slice(None), a)]
        * jnp.take_along_axis(x, (shift_idx + a)[lead + (slice(None),)], axis=-1)
        for a in range(shift_w.shape[-1])
    )


class ProjectionOperator(NamedTuple):
    """The shared factors of the line-of-sight projection, from :func:`projection_operator`."""

    interp: jax.Array
    bessel: jax.Array
    bessel_e: jax.Array
    lnk_weights: jax.Array


def projection_operator(ells, k_values, k_fine, tau_save, tau0_ref, dtype=jnp.float32):
    """Return the shared factors of the line-of-sight projection, cached per grids.

    ``interp`` is the ``(n_k_fine, n_k)`` natural-cubic-spline matrix in ``ln k``
    (the spline applied to the identity, so it reproduces
    :class:`~discoeb.spline_interpolation.spline_interpolation` to rounding);
    ``bessel`` is the ``(n_k_fine, n_ell, 3, n_chi)`` table of ``j_l``, ``j_l'``
    and ``j_l''`` at ``x = k chi`` on the shared grid ``chi = tau0_ref - tau_save``
    with the trapezoid weights of the ``tau`` integral folded in, so that

        ``Theta_l(k) = sum_s sum_j A[s, k, j] bessel[k, l, s, j]``

    for the three shifted, refined source channels ``A``; ``bessel_e`` is the
    ``(n_k_fine, n_ell, n_chi)`` table of the E-mode kernel
    ``sqrt((l+2)!/(l-2)!) j_l(x) / x^2`` with the same weights, contracted
    against the third channel alone; ``lnk_weights`` are the trapezoid weights
    of the ``C_l`` quadrature in ``ln k``.
    """

    return _projection_operator(
        np.asarray(ells, dtype=np.int64),
        np.asarray(k_values, dtype=np.float64),
        np.asarray(k_fine, dtype=np.float64),
        np.asarray(tau_save, dtype=np.float64),
        float(tau0_ref),
        dtype,
    )


@array_cache
def _projection_operator(ells, k_values, k_fine, tau_save, tau0_ref, dtype):
    interp = spline_interpolation(
        jnp.log(jnp.asarray(k_values)), jnp.eye(len(k_values))
    ).evaluate(jnp.log(jnp.asarray(k_fine)))

    chi = tau0_ref - tau_save
    x = jnp.asarray(k_fine)[:, None] * jnp.asarray(chi)[None, :]
    # x_max rounded up so that nearby grids share one table
    x_max = float(np.max(k_fine) * np.max(chi)) + 2.0
    inv_dx, n_x, jl_tab, jl1_tab, jlx2_tab = _bessel_tables(
        ells, float(np.ceil(x_max / 250.0) * 250.0), 0.03
    )
    weights = jnp.asarray(trapezoid_weights(tau_save))
    # x is the same for every multipole and table: interpolate it once
    idx, frac = _uniform_stencil(x, inv_dx, n_x)
    inv_x = jnp.where(x > 1.0e-30, 1.0 / jnp.where(x > 1.0e-30, x, 1.0), 0.0)

    def interpolate(vals):
        return (1.0 - frac) * vals[idx] + frac * vals[idx + 1]

    def rows(i, ell):
        jl = interpolate(jl_tab[i])
        jl1 = interpolate(jl1_tab[i])
        jld = ell * inv_x * jl - jl1
        jldd = -2.0 * inv_x * jld + (ell * (ell + 1.0) * inv_x * inv_x - 1.0) * jl
        return jnp.stack([jl, jld, jldd]) * weights

    def row_e(i, ell):
        prefactor = math.sqrt((ell + 2.0) * (ell + 1.0) * ell * (ell - 1.0))
        return interpolate(jlx2_tab[i]) * (prefactor * weights)

    # Stored as (n_k_fine, n_ell, 3, n_chi), the operand layout of the
    # contraction in :func:`theta_ell_batch`: with (n_ell, 3, n_k_fine, n_chi)
    # XLA transposes the table (gigabytes at inference grids) at every call.
    bessel = jnp.stack(
        [
            jnp.transpose(rows(i, float(ell)), (1, 0, 2)).astype(dtype)
            for i, ell in enumerate(ells)
        ],
        axis=1,
    )
    bessel_e = jnp.stack(
        [row_e(i, float(ell)).astype(dtype) for i, ell in enumerate(ells)], axis=1
    )

    lnk_weights = trapezoid_weights(np.log(k_fine))
    return ProjectionOperator(
        interp.astype(dtype), bessel, bessel_e, jnp.asarray(lnk_weights)
    )


@functools.partial(jax.jit, static_argnames=("k_chunk",))
def theta_ell_batch(
    channels, tau, shift_idx, shift_w, band, interp, bessel, bessel_e, k_chunk=None
):
    """``(Theta_l, E_l)`` at ``k_fine`` of a batch of cosmologies from their source channels, as matrix products.

    ``channels`` is ``(n_c, 5, n_k, n_tau)`` on the shared save grid; ``shift_idx``
    and ``shift_w`` are each cosmology's stencil onto the shared ``chi`` grid
    (:func:`lagrange_shift_stencil`); ``band`` is the ``(n_c, n_ell, n_k_fine)``
    mask of :func:`k_band`; ``interp``, ``bessel`` and ``bessel_e`` come from
    :func:`projection_operator`. Returns the temperature and E-mode transfer
    functions, each ``(n_c, n_ell, n_k_fine)`` in float64.

    The fine wave modes are contracted ``k_chunk`` at a time (all at once by
    default), each chunk for every cosmology: the Bessel table, gigabytes at
    production sizes, is then read once a call whatever the batch, and only a
    chunk's fine-``k`` sources are ever held. Chunking changes no sum.
    """

    dtype = bessel.dtype
    sources = assemble_sources(channels, tau[None, None, :])  # (c, 3, k, tau)
    shifted = shift_onto_chi(sources, shift_idx, shift_w).astype(dtype)
    n_fine = interp.shape[0]
    step = n_fine if k_chunk is None else max(1, min(int(k_chunk), n_fine))
    thetas, thetas_e = [], []
    for f0 in range(0, n_fine, step):
        f1 = min(f0 + step, n_fine)
        # k_fine leading, the batch axis of the Bessel contraction: with (c, 3,
        # k_fine, chi) XLA transposed the refined array first; (k_fine, l, c)
        # out is the batched GEMM's own layout
        refined = jnp.einsum("fk,cskt->fcst", interp[f0:f1], shifted)
        thetas.append(jnp.einsum("fcst,flst->flc", refined, bessel[f0:f1]))
        thetas_e.append(jnp.einsum("fct,flt->flc", refined[:, :, 2], bessel_e[f0:f1]))
    theta = jnp.transpose(jnp.concatenate(thetas), (2, 1, 0))  # (c, l, k_fine)
    theta_e = jnp.transpose(jnp.concatenate(thetas_e), (2, 1, 0))
    return (
        jnp.where(band, theta, 0.0).astype(jnp.float64),
        jnp.where(band, theta_e, 0.0).astype(jnp.float64),
    )


def projection_k_chunk(n_cosmologies, n_k_fine, n_tau, dtype) -> int:
    """Fine wave modes per chunk of :func:`theta_ell_batch` within :data:`PROJECTION_BUDGET_BYTES`."""

    per_mode = 3 * n_cosmologies * n_tau * jnp.dtype(dtype).itemsize * 2
    return max(1, min(n_k_fine, int(PROJECTION_BUDGET_BYTES // max(per_mode, 1))))


class TransferBatch(NamedTuple):
    """The cosmology-dependent half of a batch's spectra, from :func:`~discocmb.main.compute_transfer_batch`.

    Everything the perturbation solve and the projection produce before the
    primordial spectrum enters: the spectra are quadratic forms of these
    tables with ``P_R(k)`` (:func:`~discocmb.spectra.spectra_from_transfer`), so a change of
    ``A_s`` or ``n_s`` alone needs no new solve.
    """

    theta: jax.Array  # (n_c, n_ell, n_k_fine) temperature transfer Theta_l(k), float64, zero outside each l's k band
    theta_e: jax.Array  # (n_c, n_ell, n_k_fine) E-mode transfer
    theta_psi: (
        jax.Array | None
    )  # (n_c, n_ell_psi_exact, n_k) lensing-potential transfer at the solved modes
    limber: (
        jax.Array | None
    )  # (n_c, n_ell_psi_limber, n_chi) the Limber integrand without P_R
    T_cmb: jax.Array  # (n_c,)


class TransferGrids(NamedTuple):
    """The cosmology-independent arrays :func:`~discocmb.spectra.spectra_from_transfer` contracts a :class:`TransferBatch` with."""

    ells: np.ndarray
    k_fine: jax.Array
    lnk_weights: jax.Array
    k_pivot: float
    ells_psi: np.ndarray | None  # the lensing-potential multipoles, exact ones first
    k_values: jax.Array | None  # the solved modes, ascending
    k_limber: jax.Array | None  # (n_ell_limber, n_chi) k_chi of the Limber multipoles
