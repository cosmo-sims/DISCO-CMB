# How it works

`discocmb.main` runs four steps, each in its own module. `discocmb.grids`
builds the wave-number and time grids they share from a fiducial cosmology.

1. **Sources** (`discocmb.sources`). DISCO-EB solves the perturbations of
   every wave mode of every cosmology in one GPU launch. A component appended
   to its model evaluates the line-of-sight sources inside the kernel and
   keeps six numbers per save time instead of the full state.
2. **Line-of-sight integration** (`discocmb.transfer`). Batched matrix
   products shift the sources onto a shared grid of comoving distance, refine
   them in $\ln k$ and contract them with tabulated spherical Bessel
   functions. This gives the transfer functions $\Theta_\ell(k)$ and
   $E_\ell(k)$.
3. **Power spectra** (`discocmb.spectra`). The transfer functions and the
   primordial spectrum give the unlensed spectra,
   $C_\ell^{XY} = 4\pi \int \mathrm{d}\ln k \, P_\mathcal{R}(k) X_\ell(k) Y_\ell(k)$.
4. **Lensing** (`discocmb.lensing`). The saved Weyl potential gives the
   lensing potential spectrum. The full-sky correlation-function method then
   lenses TT, TE and EE and produces BB.

Build the grids once with `cmb_grids` and pass them to every batch. The
Bessel tables then stay cached, and a fresh batch of 128 cosmologies takes
under a second.
