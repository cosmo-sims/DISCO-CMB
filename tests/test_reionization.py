"""The CMB spectra with reionization against CLASS.

The unlensed TT, TE and EE spectra of a batch of cosmologies with different
optical depths match CLASS's ``reio_camb`` to the gates of
``tests/test_cmb.py``, from the reionization bumps at low ``l`` (the EE bump
is the signature of reionization) to the ``exp(-2 tau_reio)`` suppression of
the acoustic peaks. DISCO-EB's ``tests/test_reionization.py`` checks the
thermal history itself.
"""

import numpy as np
import pytest
from cosmologies import _perturbed_cosmologies
from test_cmb import _assert_matches_class, _cuda_available

# Multipoles of the spectrum check: the reionization bumps (l < 20) and the peaks.
TT_ELLS = np.asarray(
    [2, 3, 4, 6, 8, 10, 15, 20, 30, 40, 80, 120, 180, 240, 320, 500, 800, 1200],
    dtype=np.int64,
)


def _class(param, extra):
    """A computed CLASS instance for ``param`` with CAMB-style reionization at its ``tau_reio``."""

    Class = pytest.importorskip("classy").Class
    h = param["H0"] / 100.0
    c = Class()
    c.set(
        {
            "h": h,
            "omega_b": param["Omegab"] * h**2,
            "omega_cdm": (param["Omegam"] - param["Omegab"]) * h**2,
            "A_s": param["A_s"],
            "n_s": param["n_s"],
            "k_pivot": param["k_p"],
            "YHe": param["YHe"],
            "T_cmb": param["Tcmb"],
            "N_ur": param["Neff"],
            "N_ncdm": 0,
            "Omega_k": 0.0,
            "recombination": "recfast",
            "reio_parametrization": "reio_camb",
            "tau_reio": param["tau_reio"],
            **extra,
        }
    )
    c.compute()
    return c


@pytest.mark.skipif(not _cuda_available(), reason="numba-CUDA requires a CUDA GPU")
def test_power_spectra_with_reionization_match_class():
    """Unlensed TT, TE and EE D_l of a batch with tau_reio from 0.04 to 0.07 against CLASS reio_camb."""

    import jax

    from discocmb.grids import cmb_grids, cmb_k_grid
    from discocmb.main import compute_cl_power_spectrum_batch

    jax.config.update("jax_enable_x64", True)
    cosmologies = [
        {**c, "tau_reio": t}
        for c, t in zip(_perturbed_cosmologies(4), np.linspace(0.04, 0.07, 4))
    ]
    k_values = cmb_k_grid(cosmologies[0], n=128, mode="ode", k_min=1.0e-5, k_max=0.5)
    grids = cmb_grids(
        cosmologies[0], k_values=k_values, ells=TT_ELLS, n_save=1000, n_k_fine=1500
    )
    _, _, dl = compute_cl_power_spectrum_batch(cosmologies, grids=grids)
    dl = jax.block_until_ready(dl)

    dl_class = []
    for cosmology in cosmologies:
        c = _class(
            cosmology,
            {"output": "tCl,pCl", "modes": "s", "lensing": "no", "l_max_scalars": 2500},
        )
        try:
            raw = c.raw_cl(int(TT_ELLS.max()) + 1)
        finally:
            c.struct_cleanup()
            c.empty()
        ell = raw["ell"]
        factor = ell * (ell + 1.0) * (cosmology["Tcmb"] * 1.0e6) ** 2 / (2.0 * np.pi)
        dl_class.append(
            {name: (raw[name] * factor)[TT_ELLS] for name in ("tt", "te", "ee")}
        )
    _assert_matches_class(
        dl,
        dl_class,
        TT_ELLS,
        who=lambda i: f"cosmology {i} (tau_reio {cosmologies[i]['tau_reio']:.3f})",
    )
