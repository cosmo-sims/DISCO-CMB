"""The cosmologies the tests share, as DISCO-EB's ``tests/test_perturbations.py`` defines them.

Kept identical to DISCO-EB's so the CMB spectra here are checked on the same
batches as its matter power spectrum.
"""

import time

import numpy as np
import pytest

REFERENCE_COSMOLOGY = {
    "Omegam": (0.02238280 + 0.1201075) / 0.6732117**2,
    "Omegab": 0.02238280 / 0.6732117**2,
    "w_DE_0": -1.0,
    "w_DE_a": 0.0,
    "cs2_DE": 1.0,
    "Omegak": 0.0,
    "A_s": 2.100549e-9,
    "n_s": 0.9660499,
    "H0": 67.32117,
    "Tcmb": 2.7255,
    "YHe": 0.2454006,
    "Neff": 3.046,
    "Nmnu": 0.0,
    "mnu": 0.0,
    "k_p": 0.05,
}

DENSITY_NAMES = ("grhog", "grhornomass", "grhoc", "grhob", "grhov")
BATCH_SIZE_CASES = [pytest.param(1, id="N1"), pytest.param(128, id="N128")]


def _perturbed_cosmologies(n):
    """Return deterministic nearby flat LCDM parameter dictionaries."""

    if n == 1:
        return [REFERENCE_COSMOLOGY.copy()]
    phase = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    cosmologies = []
    for ph in phase:
        param = REFERENCE_COSMOLOGY.copy()
        omega_b_h2 = 0.02238280 * (1.0 + 0.006 * ph)
        omega_c_h2 = 0.1201075 * (1.0 - 0.008 * ph)
        h = 0.6732117 * (1.0 + 0.004 * np.sin(np.pi * ph))
        param.update(
            {
                "Omegam": (omega_b_h2 + omega_c_h2) / h**2,
                "Omegab": omega_b_h2 / h**2,
                "H0": 100.0 * h,
                "YHe": REFERENCE_COSMOLOGY["YHe"]
                * (1.0 + 0.003 * np.cos(0.5 * np.pi * ph)),
                "Tcmb": REFERENCE_COSMOLOGY["Tcmb"] * (1.0 + 0.0015 * ph),
                "Neff": REFERENCE_COSMOLOGY["Neff"]
                * (1.0 + 0.002 * np.sin(2.0 * np.pi * ph)),
            }
        )
        cosmologies.append(param)
    return cosmologies


def _successive_batches(n_batches, n_cosmologies):
    """``n_batches`` batches of ``n_cosmologies`` distinct nearby cosmologies.

    The ``n_batches * n_cosmologies`` cosmologies of :func:`_perturbed_cosmologies`
    dealt out in turn, so every batch spans the whole parameter range the
    single-batch tests' 128 do and takes comparable solver steps. The pattern
    of an inference loop: batch after batch of fresh parameters, all of one
    shape.
    """

    cosmologies = _perturbed_cosmologies(n_batches * n_cosmologies)
    return [cosmologies[i::n_batches] for i in range(n_batches)]


def _warmup_cosmologies(n):
    """``n`` cosmologies distinct from :func:`_perturbed_cosmologies` of the same ``n``.

    What a timing warms up on so the timed call, on ``_perturbed_cosmologies(n)``
    itself, finds its compiled programs and nothing of its own in the caches:
    it then pays everything a fresh batch pays, the thermodynamics included.
    """

    return _perturbed_cosmologies(n + 1)[1:]


def _wall_time(run):
    """``(seconds, result)`` of one ``run()``."""

    start = time.perf_counter()
    result = run()
    return time.perf_counter() - start, result
