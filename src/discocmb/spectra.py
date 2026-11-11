"""The unlensed CMB power spectra from the transfer functions.

The angular power spectra follow from the transfer functions of
:mod:`discocmb.transfer` and the primordial curvature spectrum,

    ``C_l^{XY} = 4 pi int dln k  P_R(k) X_l(k) Y_l(k)``,   ``P_R = A_s (k/k_p)^{n_s - 1}``

with ``X, Y`` the temperature ``Theta_l`` or the E-mode ``E_l``, and are
reported as ``D_l = l(l+1) C_l / 2pi`` in ``muK^2`` (:func:`dl_power_spectrum`).
The three spectra are returned together as a :class:`Spectra` (``tt, te,
ee``); the lensed TE, EE and BB follow in :mod:`discocmb.lensing`.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .lensing import lensing_potential_cl_from_transfer
from .transfer import TransferBatch, TransferGrids, primordial_spectrum


class Spectra(NamedTuple):
    """The scalar CMB spectra of one computation: temperature, cross and E-mode.

    Each field is an array over the multipoles (with a leading batch axis from
    :func:`~discocmb.main.compute_cl_power_spectrum_batch`); as a tuple it stacks and maps like
    one, ``jnp.stack(spectra)`` giving ``(3, ...)`` in the order ``tt, te, ee``.
    """

    tt: jax.Array
    te: jax.Array
    ee: jax.Array


@jax.jit
def cl_from_theta(theta, lnk_weights, n_s, A_s, k_pivot, k_fine, theta2=None):
    """``C_l = 4 pi int dln k P_R(k) Theta_l(k) Theta2_l(k)`` for transfer functions of shape ``(n_c, n_ell, n_k_fine)``.

    The auto spectrum of ``theta`` by default; the cross spectrum with ``theta2``.
    """

    primordial = primordial_spectrum(k_fine, n_s, A_s, k_pivot)
    product = theta**2 if theta2 is None else theta * theta2
    return 4.0 * jnp.pi * jnp.einsum("clf,cf,f->cl", product, primordial, lnk_weights)


def spectra_from_theta(
    theta, theta_e, lnk_weights, n_s, A_s, k_pivot, k_fine
) -> Spectra:
    """The ``(tt, te, ee)`` :class:`Spectra` of the two transfer functions of :func:`~discocmb.transfer.theta_ell_batch`."""

    args = (lnk_weights, n_s, A_s, k_pivot, k_fine)
    return Spectra(
        cl_from_theta(theta, *args),
        cl_from_theta(theta, *args, theta2=theta_e),
        cl_from_theta(theta_e, *args),
    )


def dl_power_spectrum(cl, ells, T_cmb):
    """Return ``D_l = l(l+1) C_l / 2pi`` in ``muK^2``."""

    ell = jnp.asarray(ells, dtype=jnp.float64)
    return ell * (ell + 1.0) * jnp.asarray(cl) / (2.0 * jnp.pi) * T_cmb**2 * 1.0e12


@jax.jit
def spectra_from_transfer(transfer: TransferBatch, tgrids: TransferGrids, n_s, A_s):
    """``(C_l, D_l, C_l^{psi psi} or None)`` of a :class:`~discocmb.transfer.TransferBatch` for primordial spectra ``A_s (k / k_pivot)^(n_s - 1)``, ``(n_c,)`` each."""

    cl = spectra_from_theta(
        transfer.theta,
        transfer.theta_e,
        tgrids.lnk_weights,
        n_s,
        A_s,
        tgrids.k_pivot,
        tgrids.k_fine,
    )
    ells_d = jnp.asarray(tgrids.ells, dtype=jnp.float64)
    dl = Spectra(*(dl_power_spectrum(c, ells_d, transfer.T_cmb[:, None]) for c in cl))
    if transfer.theta_psi is None:
        return cl, dl, None
    cl_pp = lensing_potential_cl_from_transfer(
        transfer.theta_psi,
        transfer.limber,
        tgrids.chi,
        tgrids.ells_limber,
        tgrids.k_values,
        n_s,
        A_s,
        tgrids.k_pivot,
    )
    return cl, dl, cl_pp
