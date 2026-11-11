# DISCO-CMB

CMB angular power spectra for batches of cosmologies on the GPU, from the
Einstein-Boltzmann solve of [DISCO-EB](https://github.com/ohahn/DISCO-EB):

- unlensed TT, TE and EE by line-of-sight integration, with the sources
  evaluated inside DISCO-EB's perturbation kernel (`discocmb.sources`,
  `discocmb.transfer`, `discocmb.spectra`);
- the lensing potential and the lensed TT, TE, EE and BB by the full-sky
  correlation-function method (`discocmb.lensing`);
- derivatives of the spectra through JAX (`cl_power_spectrum_jax`).

DISCO-CMB uses DISCO-EB through its interface, the `discoeb` package
itself. It was split out of DISCO-EB, whose history of these modules it
keeps.

```python
from discocmb import cmb_grids, cmb_k_grid, compute_cl_power_spectrum_batch

grids = cmb_grids(fiducial, k_values=cmb_k_grid(fiducial, n=128), n_save=1000)
ells, _, (tt, te, ee) = compute_cl_power_spectrum_batch(cosmologies, grids=grids)
```

See `examples/cmb_power_spectra_batch.py` and `AGENTS.md` for setup, tests
and the design. Licensed under the GPLv3, as DISCO-EB.
