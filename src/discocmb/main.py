"""CMB power spectra end to end, for batches of cosmologies.

The pipeline, each step in its own module:

1. **Sources** (:mod:`discocmb.sources`): one DISCO-EB launch
   (:func:`~discoeb.solve_eb`, through :func:`~discocmb.sources.solve_sources`)
   integrates the perturbation hierarchy of every wave mode of every
   cosmology with the :class:`~discocmb.sources.LineOfSightSources` component
   appended, keeping the source channels at every save time; they are then
   weighted by the visibility on the device.
2. **Line-of-sight integration** (:mod:`discocmb.transfer`): the sources are
   shifted onto a shared ``chi`` grid, refined in ``ln k`` and projected
   against the Bessel tables into ``Theta_l(k)`` and ``E_l(k)``.
3. **Power spectra** (:mod:`discocmb.spectra`): ``C_l`` and ``D_l`` of TT, TE
   and EE from the transfer functions and the primordial spectrum.
4. **Lensing** (:mod:`discocmb.lensing`): on request, ``C_l^{psi psi}`` from
   the saved Weyl potential; :func:`discocmb.lensing.lensed_spectra` then
   lenses the spectra.

The grids of every step come from :mod:`discocmb.grids`.
"""

from __future__ import annotations

import discoeb as eb
import jax.numpy as jnp
import numpy as np

from .grids import CMBGrids, cmb_grids
from .lensing import lensing_operators, lensing_potential_transfer
from .sources import channels_from_solution, solve_sources
from .spectra import Spectra, spectra_from_transfer
from .transfer import (
    TransferBatch,
    TransferGrids,
    k_band,
    lagrange_shift_stencil,
    projection_k_chunk,
    projection_operator,
    theta_ell_batch,
)


def compute_transfer_batch(
    cosmologies,
    *,
    grids: CMBGrids | None = None,
    k_values: np.ndarray | None = None,
    ells: np.ndarray | None = None,
    n_save: int = 1000,
    n_k_fine: int = 1500,
    k_fine_values: np.ndarray | None = None,
    projection_dtype=jnp.float32,
    lensing_potential_ells: np.ndarray | None = None,
    **solve_kwargs,
):
    """Return ``(TransferBatch, TransferGrids)`` for a batch of cosmologies: the spectra before the primordial spectrum.

    The first half of :func:`compute_cl_power_spectrum_batch`, which is this
    followed by :func:`~discocmb.spectra.spectra_from_transfer`; see it for the arguments.
    """

    cosmologies = tuple(eb.as_cosmology(c) for c in cosmologies)
    tau0_all = eb.conformal_ages(cosmologies)
    if grids is None:
        grids = cmb_grids(
            cosmologies[0],
            k_values=k_values,
            ells=ells,
            n_save=n_save,
            n_k_fine=n_k_fine,
            k_fine_values=k_fine_values,
            tau_end=float(np.max(tau0_all)),
        )
    k_values, ells, k_fine, tau_save = grids
    tau0_ref = grids.tau_end
    if np.max(tau0_all) > tau0_ref * (1.0 + 1.0e-12):
        raise ValueError(
            f"a cosmology's tau0 = {np.max(tau0_all):.1f} Mpc lies past the save grid's "
            f"end {tau0_ref:.1f} Mpc: build the grids with a larger tau_end_margin"
        )

    # 1. Sources: one DISCO-EB launch keeps the source channels at every save.
    # The kept sources are part of the kernel-cache key; like the right-hand
    # side they are compiled once per layout, and nothing per batch is uploaded.
    k_sorted = np.sort(np.asarray(k_values, dtype=np.float64))
    solution = solve_sources(cosmologies, k_sorted, tau_save, **solve_kwargs)
    # (n_cosmo, 6, n_k, n_tau), then the visibility weighting
    chi_star, channels = channels_from_solution(
        solution.values, jnp.asarray(tau_save), jnp.asarray(tau0_all)
    )
    del solution

    # 2. Line-of-sight integration onto the shared chi grid.
    interp, bessel, bessel_e, lnk_weights = projection_operator(
        ells, k_sorted, k_fine, tau_save, tau0_ref, projection_dtype
    )
    ells_d = jnp.asarray(ells, dtype=jnp.float64)
    k_fine_d = jnp.asarray(k_fine)
    tau_d = jnp.asarray(tau_save)
    # Everything per cosmology below is one host array and one device transfer
    # for the whole batch: the shifts onto the shared chi grid, the k bands, the
    # primordial amplitudes. A per-cosmology loop of eager ``jnp`` arithmetic
    # here cost 80 ms of host time a batch (and, while the kernel was in
    # flight, waited for it).
    chi_max = float(tau0_ref - tau_save[0])
    shift_idx, shift_w = lagrange_shift_stencil(tau_save, tau0_all - tau0_ref)
    shift_idx, shift_w = jnp.asarray(shift_idx), jnp.asarray(shift_w)
    bands = k_band(ells_d, k_fine_d, chi_max, chi_star)
    T_cmb = jnp.asarray([float(c.T_cmb) for c in cosmologies])
    k_pivot = float(cosmologies[0].k_pivot)

    # The whole batch is one call, chunked over the fine wave modes so the
    # Bessel table is read once and a chunk's sources fit the memory budget;
    # chunking over cosmologies instead read the table once per group (60 ms
    # a batch of 128 against 37).
    theta, theta_e = theta_ell_batch(
        channels,
        tau_d,
        shift_idx,
        shift_w,
        bands,
        interp,
        bessel,
        bessel_e,
        k_chunk=projection_k_chunk(
            len(cosmologies), len(k_fine), len(tau_save), projection_dtype
        ),
    )
    if lensing_potential_ells is None:
        transfer = TransferBatch(theta, theta_e, None, None, T_cmb)
        tgrids = TransferGrids(ells, k_fine_d, lnk_weights, k_pivot, None, None, None)
        return transfer, tgrids

    # 4. The lensing potential's transfer, from the saved Weyl potential.
    ells_pp = np.asarray(lensing_potential_ells, dtype=np.int64)
    operator, limber_weights, k_chi = lensing_operators(
        ells_pp, k_sorted, tau_save, tau0_ref, projection_dtype
    )
    theta_psi, limber = lensing_potential_transfer(
        channels[:, 3],
        shift_idx,
        shift_w,
        jnp.asarray(tau0_ref - tau_save),
        chi_star,
        operator,
        limber_weights,
        k_chi,
    )
    transfer = TransferBatch(theta, theta_e, theta_psi, limber, T_cmb)
    tgrids = TransferGrids(
        ells, k_fine_d, lnk_weights, k_pivot, ells_pp, jnp.asarray(k_sorted), k_chi
    )
    return transfer, tgrids


def compute_cl_power_spectrum_batch(
    cosmologies,
    *,
    grids: CMBGrids | None = None,
    k_values: np.ndarray | None = None,
    ells: np.ndarray | None = None,
    n_save: int = 1000,
    n_k_fine: int = 1500,
    k_fine_values: np.ndarray | None = None,
    projection_dtype=jnp.float32,
    lensing_potential_ells: np.ndarray | None = None,
    **solve_kwargs,
):
    """Return ``(ells, C_l, D_l)`` for a batch of cosmologies, each a :class:`~discocmb.spectra.Spectra` of ``(N, n_ell)``.

    ``C_l`` holds the unlensed scalar ``tt``, ``te`` and ``ee`` spectra
    (dimensionless, relative to ``T_cmb``) and ``D_l = l(l+1) C_l / 2pi`` the
    same in ``muK^2``. With ``lensing_potential_ells`` the lensing potential
    spectrum ``C_l^{psi psi}`` at those multipoles (:mod:`discocmb.lensing`) is
    returned as a fourth element, ``(N, n_ell_psi)``, from the Weyl potential
    the same launch recorded; :func:`discocmb.lensing.lensed_spectra` turns the
    two into the lensed TT, TE, EE and BB.


    One modax launch integrates the perturbation hierarchy of every wave mode of
    every cosmology, with the sources evaluated at the save times by the save
    hook; the launch returns ``5 n_save`` numbers per trajectory and no history,
    so nothing bounds the batch but the kernel's own memory. The sources are
    then refined in ``ln k`` onto the fine grid and projected by
    :func:`~discocmb.transfer.theta_ell_batch` and :func:`~discocmb.spectra.spectra_from_theta`.

    ``grids`` fixes the wave modes, multipoles, fine ``k`` grid and save grid
    (:func:`~discocmb.grids.cmb_grids`); built once from a fiducial cosmology, it makes the
    projection operator shared by every batch of an inference loop. Without
    it, the grids are shaped by the batch's first cosmology from the remaining
    keyword arguments, with the save grid ending at the largest ``tau0`` of the
    batch (each cosmology's own ``tau0`` is respected through zeroed sources
    past it), and the operator is rebuilt for each batch whose grids differ.

    ``projection_dtype`` is the arithmetic of the projection's matrix products:
    ``float32`` (default, 2.6e-4 in ``D_l``) or ``float64`` (ten times slower
    on consumer GPUs). ``solve_kwargs`` reach the solver (``rtol``, ``atol``,
    ``first_step``, ``max_steps``).
    """

    cosmologies = tuple(eb.as_cosmology(c) for c in cosmologies)
    transfer, tgrids = compute_transfer_batch(
        cosmologies,
        grids=grids,
        k_values=k_values,
        ells=ells,
        n_save=n_save,
        n_k_fine=n_k_fine,
        k_fine_values=k_fine_values,
        projection_dtype=projection_dtype,
        lensing_potential_ells=lensing_potential_ells,
        **solve_kwargs,
    )
    # 3. Power spectra (and C_l^{psi psi}) with the primordial spectrum.
    n_s = jnp.asarray([float(c.n_s) for c in cosmologies])
    A_s = jnp.asarray([float(c.A_s) for c in cosmologies])
    cl, dl, cl_pp = spectra_from_transfer(transfer, tgrids, n_s, A_s)
    if lensing_potential_ells is None:
        return tgrids.ells, cl, dl
    return tgrids.ells, cl, dl, cl_pp


def cl_power_spectrum_jax(
    cosmology,
    densities=None,
    thermal=None,
    *,
    grids: CMBGrids | None = None,
    k_values: np.ndarray | None = None,
    ells: np.ndarray | None = None,
    n_save: int = 1000,
    n_k_fine: int = 1500,
    k_fine_values: np.ndarray | None = None,
    projection_dtype=jnp.float64,
    **solve_kwargs,
):
    """``(ells, C_l, D_l)`` of one cosmology as JAX arrays, differentiable in the cosmology.

    The counterpart of :func:`discoeb.perturbations.matter_power_spectrum_jax`
    for the unlensed spectra (a :class:`~discocmb.spectra.Spectra` each, ``tt, te, ee`` of
    shape ``(n_ell,)``). ``densities`` (the five background densities
    ``grhog, grhornomass, grhoc, grhob, grhov``) and ``thermal`` (``YHe,
    Tcmb, H0, Omegab``) -- together the Einstein-Boltzmann part's directed
    inputs (:data:`discoeb.DIRECTED_NAMES`) -- default to the cosmology's
    own; given as JAX values, the spectra are differentiable in them. The
    derivative is formed in the kernel: :func:`~discocmb.sources.solve_sources` keeps the
    sources and, for a tangent, their derivatives along each input, by the
    forward sensitivities of both blocks of the hierarchy, so the
    recombination history is differentiated along with the hierarchy and the
    densities enter again through the source algebra; no history is stored.
    The grids, the save grid, ``tau0`` and the reionization redshift are held
    fixed. It is :func:`compute_transfer_batch` and
    :func:`~discocmb.spectra.spectra_from_transfer` for a batch of one. The solve uses :func:`~discocmb.sources.polarization_error_weights` unless
    ``error_weights`` is given. The projection defaults to ``float64``: the
    ``float32`` matrix products cost 2.6e-4 in ``D_l``, which a derivative of
    the spectrum cannot afford.
    """

    cosmology = eb.as_cosmology(cosmology)
    inputs = None
    if densities is not None or thermal is not None:
        own = jnp.asarray(eb.directed_inputs(cosmology)[0])
        inputs = jnp.concatenate(
            [
                own[:5] if densities is None else jnp.asarray(densities, jnp.float64),
                own[5:] if thermal is None else jnp.asarray(thermal, jnp.float64),
            ]
        )[None]
    transfer, tgrids = compute_transfer_batch(
        (cosmology,),
        grids=grids,
        k_values=k_values,
        ells=ells,
        n_save=n_save,
        n_k_fine=n_k_fine,
        k_fine_values=k_fine_values,
        projection_dtype=projection_dtype,
        inputs=inputs,
        **solve_kwargs,
    )
    cl, dl, _ = spectra_from_transfer(
        transfer,
        tgrids,
        jnp.asarray([float(cosmology.n_s)]),
        jnp.asarray([float(cosmology.A_s)]),
    )
    return tgrids.ells, Spectra(*(c[0] for c in cl)), Spectra(*(d[0] for d in dl))
