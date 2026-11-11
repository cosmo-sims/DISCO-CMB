# AGENTS.md

DISCO-CMB computes CMB angular power spectra -- unlensed TT, TE and EE by
line-of-sight integration, the lensing potential and the lensed TT, TE, EE and
BB -- for batches of cosmologies on the GPU, from DISCO-EB's
Einstein-Boltzmann solve. It was split out of DISCO-EB (`src/discoeb/cmb.py`
and `lensing.py`, whose history this repository keeps) on 2026-11-11.

## The boundary with DISCO-EB

`discocmb` imports from DISCO-EB's interface `discoeb.eb` and the generic
`discoeb.spline_interpolation`, nothing else (`tests/test_boundary.py` checks
the import statements and that every `eb.` name used is in `eb.__all__`).
The contract is `eb.solve_eb(cosmologies, k, tau, keep=names,
extra=components)`: named quantities evaluated in the kernel at the save
times, the line-of-sight sources being the `LineOfSightSources` component
(`cmb.py`) passed as `extra`; error weights by state name; differentiable in
`DIFFERENTIABLE_PARAMETERS` or in the nine `DIRECTED_NAMES` inputs passed as
`inputs=` (`eb.directed_inputs` gives a cosmology's). Grids read
`background_history`, `density_coefficients` and `conformal_ages`. A change
here that seems to need a private DISCO-EB name should add to DISCO-EB's
`eb.py` instead. Only the white-box test
`test_sources_come_from_the_component_model` reads `discoeb.perturbations`
directly.

## Development

DISCO-EB is a uv path dependency from a checkout beside this one
(`[tool.uv.sources]`), and with it modax and numba-enzyme from theirs (see
DISCO-EB's notes; never let a `uv sync` write through a symlinked
numba-enzyme). Until this project has its own environment, run it in
the DISCO-DJ III paper scripts' venv, which has DISCO-EB, modax and
numba-enzyme editable from their checkouts and `classy`:

```bash
V="$HOME/git/papers/DISCO-DJ III: accelerated CMB inference/scripts/.venv/bin/python"
PYTHONPATH=src "$V" -m pytest tests
PYTHONPATH=src "$V" examples/cmb_power_spectra_batch.py
```

DISCO-EB's own `.venv` also works with `PYTHONPATH=src:$HOME/git/modax` (it
holds the modax 0.0.7 wheel, without `Block`), but has no `classy`, so every
CLASS comparison skips there.

The tests need a CUDA GPU and `classy` for the CLASS references; host-only:
`tests/test_boundary.py`, `tests/test_lensing.py::test_lensed_spectra_match_class_convolution`
and `tests/test_cmb.py::test_sources_come_from_the_component_model`.
`tests/cosmologies.py` holds the cosmologies shared with DISCO-EB's
`tests/test_perturbations.py`; keep the two identical. Benchmark on the GPU
only, when it is idle.

## Modules

1. **cmb.py** - Unlensed TT, TE and EE power spectra by line-of-sight integration
   - Returns `(ells, cl, dl)` with `cl` and `dl` each a `Spectra` NamedTuple
     (`tt, te, ee`); the E-mode transfer is the quadrupole source `S_j2`
     against a fourth Bessel row `sqrt((l+2)!/(l-2)!) j_l(x)/x^2` (CAMB's
     `sources(2)`), tabulated as `j_l/x^2` directly so it interpolates through
     `x = 0`. EE within 0.35 percent of CLASS and TE within 0.4 percent of
     `sqrt(TT EE)` at l = 2..1200 (gate 2 percent, `tests/test_cmb.py`)
   - The CMB solves pass `polarization_error_weights(cosmology)` -- weights by
     state name -- to `solve_eb`: the
     photon quadrupole, `E_2` and the massive-neutrino monopole, dipole and
     quadrupole of every momentum bin weighted by `clip(1/k, 1, 1e4)` in
     the error norm. At low k those moments are below the scalar
     `atol = 1e-4`, and without the weights the low multipoles of EE and TE
     are 10 to 40 percent low (a 0.3 eV neutrino's EE at l = 2 another 2
     percent); the weights slow only the low-k trajectories, which are not the
     launch's critical path, so a batch of 128 costs 10 percent more than
     unweighted (0.64 s against 0.58 s). The weight was `clip(0.1/k, 1, 1000)`
     until 2026-11-03, at no cost, but left EE at l = 2 1.2 percent low
     (grids and save grid do not move it; only the error control does), and
     the end-to-end lensing test failed its 0.5 percent gate on the l = 2 EE
     lensing correction, where lensing adds 160 percent. Scalar
     `rtol = atol = 1e-5` reaches the same accuracy at 1.8x the cost, a
     scalar `atol = 1e-8` at 6x
   - `--benchmark-disable` makes `benchmark.pedantic` call its target once
     with no warm-up round, so a test that deals its warm-up batch and its
     checked batch out of one iterator checks the wrong batch; the TT test
     reads `benchmark.disabled` and records the batch it ran
   - `test_power_spectra_derivatives` differences at a relative step of 1e-3
     with a resolvability floor of 1e-5 of `D_l`: a step-acceptance flip
     between the two shifted solves moves `D_l` by ~1e-6 of itself, which at
     the old 1e-5 step was 7e-3 of the derivative's scale once the weights
     sharpened the low-k step control (the JVP is within 3e-4 of a converged
     difference at rtol = atol = 1e-6); the dark-energy density moves EE by
     no more than that noise, so its polarization entries are not compared
   - Sources in CAMB form (monopole + ISW against `j_l`, Doppler against `j_l'`,
     quadrupole/polarization against `j_l''`), evaluated *inside* the modax
     Rodas5P launch: `solve_sources` asks `eb.solve_eb` to keep `SOURCE_KEEP`
     (`m0, m1, m2, phi, opacity, kappa`) at every save, six numbers per save
     and trajectory instead of the 50-component state, so a whole batch is
     one launch with no chunking; the visibility weighting follows on the
     device (`_channels_from_solution`). The ISW `phi'` is a centered difference of the saved potential:
     the analytic RHS derivative is exact algebra but amplifies the solver's
     state error tenfold at l=2
   - The visibility comes from a cubic spline of the tabulated opacity whose
     integral gives kappa (`visibility_on_grid`). A trapezoid of kappa' on the
     ~3 Mpc save grid was 0.2 percent too large through recombination and
     lowered every C_l by 0.5 percent -- this, not RECFAST, was the "1 percent
     discrepancy" with CLASS (the gauge-invariant photon shear matches CLASS
     to 1e-4 at z=1100; P(k) to 1e-4)
   - Projection as batched matrix products on grids shared by the batch
     (`projection_operator`, `theta_ell_batch`): sources shifted in tau onto one
     chi grid by a 4-point Lagrange stencil (each tau0 differs from the reference
     by up to ~100 Mpc), refined in ln k by a precomputed spline matrix, and
     contracted against one weighted table of `j_l`, `j_l'`, `j_l''` on the
     (k, chi) grid. 0.25 ms per cosmology in float32 (default; 2.6e-4 in D_l),
     2.3 ms in float64 (`projection_dtype`). Warm: ~0.29 s at N=1, ~0.7 s at
     N=128 (RTX 4070 SUPER); within 0.25 percent of CLASS at `reio_none` (and
     `reio_camb` with `tau_reio`) for
     l = 2..1200, gate 0.5 percent
   - `cmb_grids(fiducial, ...)` builds the `CMBGrids` (wave modes, multipoles,
     fine k grid, save grid) once; passed as `grids=` to
     `compute_cl_power_spectrum_batch` they fix the projection operator across
     batches (else it is rebuilt per batch, 0.3 s). The save grid ends
     `TAU_END_MARGIN` (1 percent) past the fiducial's tau0: every cosmology's
     tau0 must lie below it (`ValueError` otherwise), and the shift onto the
     fiducial grid costs 2.7e-4 in D_l at 1 percent, 1e-3 at 2, 4e-3 at 5.
     A fresh 128-batch then costs ~1.0 s (thermo 0.3, kernel 0.6)
   - Rejected after measurement: FFTLog projection (needs 32768 log-chi points
     to resolve recombination and is an order of magnitude more work than the
     table contraction); in-kernel `Theta_l` accumulation (must solve on the
     1500-mode fine k grid, 10 s at N=128); piecewise-in-tau solves; a saved
     history with host-side sources
   - The Bessel table is stored `(n_k_fine, n_ell, 3, n_chi)`, the operand
     layout of the contraction; the old `(n_ell, 3, n_k_fine, n_chi)` made XLA
     transpose gigabytes per call
   - Above l ~ 1500 the unlensed TT is 1 to 3 percent off CLASS (the damping
     tail is not gated; `test_power_spectra_match_class` stops at 1200)
   - The sources are the `LineOfSightSources` component appended to the
     perturbation model (`source_model(layout, thermal_history)`): `phi` from
     the sums `dgrho`, `dgq`, `dgpi`, the three CAMB-form sources from
     `polter`, `sigma`, `vb`, `g`. `solve_eb` compiles them into its keep hook,
     for the batch and for the differentiable path alike
     (`cl_power_spectrum_jax` passes the densities and thermal inputs as
     `inputs=`, and the tangent is formed in the kernel, with no history
     saved), so there is one copy of the algebra and every species a layout
     enables enters it. Gated against CLASS for flat
     LCDM with massless neutrinos (0.5 percent TT) and with one 0.3 eV
     neutrino (`test_power_spectra_match_class_with_massive_neutrinos`: TT
     within 1 percent, EE/TE within 2 percent; CLASS gets `T_ncdm =
     (4/11)^(1/3)`, our exact neutrino temperature, and `N_ur` = the massless
     `Neff`). `cl_power_spectrum_jax` builds the model without the thermal
     history, so a massive-neutrino layout fails there with a `NameError` on
     `rhonu`

2. **lensing.py** - Lensing potential and lensed TT, TE, EE, BB
   - `C_l^{psi psi}` from the saved Weyl potential (`phi` channel) against the
     lensing kernel: exact projection below `LIMBER_LMIN` = 200, Limber above
     (needs solved modes to k ~ 1 Mpc^-1 for l ~ 2500). Requested through
     `compute_cl_power_spectrum_batch(..., lensing_potential_ells=...)`, which
     then returns a fourth element
   - `lensed_spectra(tt, te, ee, pp, tables)`: full-sky correlation-function
     method in the form of CAMB's `CorrFuncFullSky` (isotropic deflection to
     all orders, anisotropic to second order in `C_gl,2`), returning lensed
     TT, TE, EE and BB; twelve Wigner d-tables from `lensing_tables(lmax)`
     built once (~600 MB at l = 3000 in float32); sums over every l, so
     `cl_on_every_ell` splines the sampled spectra. `lensed_tt` is the
     temperature-only wrapper
   - Gates (`tests/test_lensing.py`): the convolution alone reproduces CLASS's
     lensed TT and EE from CLASS's inputs to 5e-4, TE to 3e-4 of
     `sqrt(TT EE)`, BB to 1 percent; `C_l^{psi psi}` within 1 percent of
     CLASS to l = 1000; end to end, lensed TT within 5e-3 and EE/TE within
     2 percent of CLASS to l = 1500. CLASS references need
     `perturbations_sampling_stepsize = 0.01`: at the default 0.1 CLASS's own
     `C_2^{pp}` is 1.2 percent low
   - A Bessel table of ~150 multipoles to l = 3000 does not fit a 12 GB device;
     sample the acoustic peaks every 40 (as CLASS does)
