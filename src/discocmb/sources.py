"""The CMB temperature and polarization source functions.

The CMB anisotropy today is obtained by *line-of-sight integration* (Seljak &
Zaldarriaga 1996, :mod:`discocmb.transfer`) of a few local source terms built
from the perturbation state at the save times and the visibility function
``g = kappa' exp(-kappa)``:

    * ``S``    -- the intrinsic monopole (Sachs-Wolfe) and polarization
      correction, weighted by ``g``, plus the integrated Sachs-Wolfe term
      ``2 Phi' exp(-kappa)`` which accumulates along the whole line of sight;
    * ``S_j1`` -- the Doppler term ``g (sigma + v_b)``;
    * ``S_j2`` -- the quadrupole/polarization term,

integrated against ``j_l(x)``, ``j_l'(x)`` and ``j_l''(x)``, ``x = k (tau_0 -
tau)``. The E-mode transfer function needs no further source: it is ``S_j2``
against ``j_l(x) / x^2`` (CAMB's ``sources(2)``).

**Where the sources are evaluated.** Inside the Einstein-Boltzmann launch,
through :mod:`discoeb`: the sources are a :class:`LineOfSightSources`
component of the model, and :func:`solve_sources` asks
:func:`~discoeb.solve_eb` to keep them, with the opacity and the optical
depth, at every save time (:data:`SOURCE_KEEP`); the kernel evaluates them
from the dense-output state and returns six numbers per save and trajectory
instead of the ``nvar``-component history, so all cosmologies of a batch
share one launch and nothing is chunked. The ISW term's ``phi'`` is a
centered difference of the saved potential (:func:`assemble_sources`): the
analytic derivative from the right-hand side is exact algebra but amplifies
the solver's state error tenfold at ``l = 2``. The visibility ``g = kappa'
exp(-kappa)`` comes from the thermodynamics table through a cubic spline
whose integral gives ``kappa`` (:func:`visibility_on_grid`); a trapezoid of
``kappa'`` on the save grid was 0.2 percent too large through recombination
and cost 0.5 percent of ``C_l``.

**Polarization.** The polarization source is a small state component at low
``k``, below the scalar absolute tolerance of the solve, so the CMB solves
weight the photon quadrupole and ``E_2`` up in the solver's error norm in
proportion to ``1/k`` (:func:`polarization_error_weights`); without that the
low multipoles of EE and TE come out 10 to 40 percent low, at no gain in
speed, and the weight has to reach ``1e4`` at ``k ~ 1e-4`` for EE at
``l = 2`` to come within 0.2 percent of CLASS.
"""

from __future__ import annotations

import discoeb as eb
import jax
import jax.numpy as jnp
import numpy as np
from discoeb import Component

SOURCE_CHANNELS = 6
"""Numbers the solve keeps per save (:data:`SOURCE_KEEP`): ``m0, m1, m2, phi, kappa', kappa``: the three
sources before their visibility weighting, the Weyl potential, the opacity and
the optical depth from ``tau_start``; :func:`weight_channels` turns them into
``S_noisw, S_j1, S_j2, phi, exp(-kappa)``."""

WEIGHTED_CHANNELS = 5
"""Numbers per save after the weighting: ``S_noisw, S_j1, S_j2, phi, exp(-kappa)``."""


class LineOfSightSources(Component):
    """The CAMB-form temperature and polarization sources, as a component.

    Appended to the perturbation model, it provides the Weyl potential
    ``phi = -(dgrho + 3 aH dgq / k + dgpi) / (2 k^2)`` from the Einstein
    sources every species contributes to, and the three sources against
    ``j_l``, ``j_l'`` and ``j_l''`` of the module docstring, without the ISW
    term: ``g (monopole + 5/8 Pi)``, ``g (sigma + v_b)`` and ``15/8 g Pi``,
    with the visibility ``g`` given and the scattering source ``Pi``
    (``polter``) and the shear ``sigma`` from the Thomson interaction and the
    metric. One set of equations serves the device hook and the differentiable
    differentiable path (:func:`~discocmb.main.cl_power_spectrum_jax`), and any species the layout enables
    enters through the sums.
    """

    def phi(self, dgrho, dgrho_streaming, dgq, dgpi, adotoa, k, inv_k2):
        """The Weyl potential, with the radiation's density under the streaming approximation."""
        return -(dgrho + dgrho_streaming + 3.0 * dgq * adotoa / k + dgpi) * 0.5 * inv_k2

    def monopole(self, etak, k, phi, clxg):
        return -etak / k + 2.0 * phi + 0.25 * clxg

    def m0(self, monopole, polter):
        return monopole + 0.625 * polter

    def m1(self, sigma, vb):
        return sigma + vb

    def m2(self, polter):
        return 1.875 * polter


def source_model(layout):
    """The joint model of ``layout`` (hierarchy and recombination) with the :class:`LineOfSightSources` appended."""

    return eb.joint_model(layout, extra=(LineOfSightSources,))


POLARIZATION_WEIGHT_K = 1.0
"""Wave number (Mpc^-1) below which the photon quadrupole and ``E_2`` are weighted
up in the solver's error norm, by ``POLARIZATION_WEIGHT_K / k``."""

POLARIZATION_WEIGHT_MAX = 1.0e4
"""Cap of that weight."""


def polarization_error_weights(cosmology) -> dict:
    """``{state name: weight of k}``: solver error weights that resolve the small sources at low ``k``.

    The E-mode source is ``g polter`` with ``polter = pig/10 + 3/5 E_2``. At
    ``k`` well below the acoustic scale both moments are tiny through
    recombination -- of the order of the solve's scalar ``atol = 1e-4`` at
    ``k ~ 1e-3 Mpc^-1`` -- so the step-size control, which measures each
    component's error against ``atol + rtol |y|``, does not resolve them, and
    the low multipoles of EE and TE, which those modes source, came out 10 to
    40 percent low (TT, whose sources are unit-size, was unaffected).
    Weighting the photon quadrupole ``theta_2`` and ``E_2`` (``e_0``) by
    ``clip(POLARIZATION_WEIGHT_K / k, 1, POLARIZATION_WEIGHT_MAX)`` divides
    their tolerance by that factor. The trajectories it slows are the
    low-``k`` ones, which take the fewest steps of a launch: at
    ``POLARIZATION_WEIGHT_K = 1`` a batch of 128 costs 10 percent more than
    unweighted (0.64 s against 0.58 s on an RTX 4070 SUPER) and EE agrees
    with CLASS to 0.35 percent at every multipole from 2 over the batch. At
    0.1 the batch cost the same but EE at ``l = 2`` was 1.2 percent low,
    which the end-to-end lensing check amplified to 0.75 percent in the
    ``l = 2`` lensing correction (lensing adds 160 percent there); neither the
    grids nor the save grid move that multipole, only the solve's error
    control does. Tightening the scalar tolerances to 1e-5 instead reaches the
    same accuracy at 1.8 times the cost, and a scalar ``atol`` of 1e-8 at six
    times.

    The massive-neutrino monopole, dipole and quadrupole of every momentum
    bin (``psi_q_0`` .. ``psi_q_2``) get the same weight: they are as small
    at low ``k`` and enter the metric shear that generates the photon
    quadrupole, and without the weight a 0.3 eV neutrino left EE at
    ``l = 2`` 2 percent low (0.3 percent with it;
    ``test_power_spectra_match_class_with_massive_neutrinos``).

    The weights are named by the cosmology's state names
    (:func:`discoeb.state_names`), the form :func:`discoeb.solve_eb`
    takes as ``error_weights``.
    """

    def weight(k):
        return np.clip(
            POLARIZATION_WEIGHT_K / np.asarray(k), 1.0, POLARIZATION_WEIGHT_MAX
        )

    names = eb.state_names(cosmology)
    weighted = ["theta_2", "e_0"] + [
        n for n in names if n.startswith("psi_") and int(n.rsplit("_", 1)[1]) < 3
    ]
    return {n: weight for n in weighted}


# =============================================================================
# The visibility function
# =============================================================================


def _grad_last(y, x):
    """Return ``dy/dx`` along the last axis by centered differences."""

    mid = (y[..., 2:] - y[..., :-2]) / (x[..., 2:] - x[..., :-2])
    first = (y[..., 1:2] - y[..., :1]) / (x[..., 1:2] - x[..., :1])
    last = (y[..., -1:] - y[..., -2:-1]) / (x[..., -1:] - x[..., -2:-1])
    return jnp.concatenate([first, mid, last], axis=-1)


VISIBILITY_REFINEMENT = 16
"""Sub-intervals per thermodynamics-table interval when integrating the optical depth."""


def visibility_on_grid(tau_grid, opacity, tau_save, tau0):
    """Return ``(kappa', g, exp(-kappa))`` on the save grid from a tabulated opacity.

    ``kappa'`` is interpolated by a cubic spline of ``ln kappa'`` in ``ln tau``
    (the thermodynamics table is log-spaced), and the optical depth
    ``kappa(tau) = int_tau^{tau0} kappa' dtau'`` is that spline integrated on a
    grid refined :data:`VISIBILITY_REFINEMENT`-fold. Integrating ``kappa'`` by
    the trapezoid rule on the save grid itself is not good enough: ``kappa'``
    falls by a factor ``e`` every ~15 Mpc through recombination, so on a 3 Mpc
    grid the trapezoid overestimates ``kappa`` by ``(h/L)^2/12 ~ 3e-3`` where
    ``kappa ~ 1``, which lowers ``g`` by 0.3 percent and ``C_l`` by 0.5 percent
    at every multipole. Past ``tau0`` both ``g`` and ``exp(-kappa)`` are zero,
    which zeroes every source there (the frozen state of a trajectory that has
    reached its own end is not physical source data).
    """

    from scipy.interpolate import CubicSpline

    tau_grid = np.asarray(tau_grid, dtype=np.float64)
    opacity = np.asarray(opacity, dtype=np.float64)
    log_spline = CubicSpline(np.log(tau_grid), np.log(opacity))

    n = len(tau_grid)
    fine = np.exp(
        np.linspace(
            np.log(tau_grid[0]), np.log(tau0), VISIBILITY_REFINEMENT * (n - 1) + 1
        )
    )
    kappa_prime_fine = np.exp(log_spline(np.log(fine)))
    kappa_fine = np.concatenate(
        [
            np.cumsum(
                (0.5 * (kappa_prime_fine[1:] + kappa_prime_fine[:-1]) * np.diff(fine))[
                    ::-1
                ]
            )[::-1],
            [0.0],
        ]
    )

    live = tau_save <= tau0
    tau_live = np.minimum(tau_save, tau0)
    kappa_prime = np.where(live, np.exp(log_spline(np.log(tau_live))), 0.0)
    kappa = np.interp(tau_live, fine, kappa_fine)
    emk = np.where(live, np.exp(-kappa), 0.0)
    return kappa_prime, kappa_prime * emk, emk


# =============================================================================
# Visibility weighting and the ISW term
# =============================================================================


def weight_channels(channels, tau_save, tau0):
    """The hook's six channels -> the five weighted ones ``S_noisw, S_j1, S_j2, phi, exp(-kappa)``.

    ``channels`` is ``(n_cosmo, 6, n_k, n_tau)`` and ``tau0`` the cosmologies'
    conformal ages. The optical depth to today is ``kappa(tau_0) - kappa(tau)``
    with ``kappa(tau_0)`` the state at the last save (every trajectory's final
    state, frozen past its own ``tau_0``); its early-time bulk cancels in the
    difference, so ``exp(-kappa)`` is as accurate as the kernel's increments.
    Past a cosmology's own ``tau_0`` both factors are zero, which zeroes every
    source there.
    """

    m0, m1, m2, phi, kappa_prime, kappa = (
        channels[:, i] for i in range(SOURCE_CHANNELS)
    )
    kappa_end = kappa[:, :, -1:]
    live = jnp.asarray(tau_save)[None, None, :] <= jnp.asarray(tau0)[:, None, None]
    emk = jnp.where(live, jnp.exp(-(kappa_end - kappa)), 0.0)
    g = kappa_prime * emk
    return jnp.stack([g * m0, g * m1, g * m2, phi, emk], axis=1)


@jax.jit
def channels_from_solution(recorded, tau_save, tau0):
    """The kept channels ``(n_cosmo, 6, n_k, n_tau)`` -> ``(chi_star, weighted channels)``, as one program.

    :func:`weight_channels` and the distance ``chi_star = tau0 - tau_star`` to
    where each cosmology's visibility ``g = kappa' exp(-kappa)`` peaks on the
    save grid, read from the first wave mode (the recombination states are the
    same on every mode of a cosmology), so no stand-alone thermal solve is
    needed per cosmology. One jitted program: run eagerly, each of its dozen
    steps was a pass over the batch's gigabyte of sources.
    """

    weighted = weight_channels(recorded, tau_save, tau0)
    g = recorded[:, 4, 0] * weighted[:, 4, 0]  # kappa' exp(-kappa), zero past tau0
    return tau0 - tau_save[jnp.argmax(g, axis=1)], weighted


def assemble_sources(channels, tau):
    """``(S, S_j1, S_j2)`` from the five weighted channels, with the ISW term by centered differences.

    ``channels`` is ``(..., 5, n_k, n_tau)``; the returned array is ``(..., 3, n_k, n_tau)``.
    """

    s_noisw, s_j1, s_j2, phi, emk = (
        channels[..., i, :, :] for i in range(WEIGHTED_CHANNELS)
    )
    s_j0 = s_noisw + 2.0 * _grad_last(phi, jnp.asarray(tau)) * emk
    return jnp.stack([s_j0, s_j1, s_j2], axis=-3)


# =============================================================================
# The sources, kept by the Einstein-Boltzmann solve
# =============================================================================
#
# The Einstein-Boltzmann launch evaluates any quantity of its model at every
# save time from the dense-output state (discoeb.solve_eb), so the sources
# are kept in the kernel and the launch returns six numbers per save instead of
# the ``nvar``-component history; the projection proceeds on the device.


SOURCE_KEEP = ("m0", "m1", "m2", "phi", "opacity", "kappa")
"""What :func:`solve_sources` keeps at every save, in channel order: the three
CAMB-form sources of the module docstring before their visibility weighting
and without the ISW term, the Weyl potential, the opacity ``kappa'`` and the
optical depth ``kappa`` (a state of the thermal block). ``phi'`` is
deliberately not kept: differentiating the right-hand side analytically at a
save is exact algebra but amplifies the solver's state error, since the
late-time ``phi'`` is a small residual of large cancelling terms; a difference
of saved potentials cancels that error, so :func:`assemble_sources` differences
``phi`` across saves."""


def solve_sources(cosmologies, k_values, tau_save, *, inputs=None, **solve_kwargs):
    """The source channels of a batch, ``EBSolution.values`` of ``(n_cosmo, 6, n_k, n_tau)``.

    One :func:`discoeb.solve_eb` launch keeping :data:`SOURCE_KEEP`, with
    :class:`LineOfSightSources` appended to the model; ``tau_save`` starts at
    ``TAU_START``. Unless ``solve_kwargs`` carries ``error_weights``, the
    launch uses :func:`polarization_error_weights`. ``inputs`` are the
    Einstein-Boltzmann part's directed inputs, given to differentiate with
    respect to them (:func:`~discocmb.main.cl_power_spectrum_jax`).
    """

    cosmologies = tuple(cosmologies)
    if solve_kwargs.get("error_weights") is None:
        solve_kwargs["error_weights"] = polarization_error_weights(cosmologies[0])
    return eb.solve_eb(
        cosmologies,
        k_values,
        tau_save,
        keep=SOURCE_KEEP,
        extra=(LineOfSightSources,),
        inputs=inputs,
        **solve_kwargs,
    )
