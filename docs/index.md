# Quickstart

```{include} ../README.md
:start-after: <!-- start:overview -->
:end-before: <!-- end:overview -->
```

## Installation

```{include} ../README.md
:start-after: <!-- start:install -->
:end-before: <!-- end:install -->
```

## Your first spectra

```{include} ../README.md
:start-after: <!-- start:quickstart -->
:end-before: <!-- end:quickstart -->
```

`cl` and `dl` each hold the `tt`, `te` and `ee` spectra with shape
`(n_cosmologies, n_ell)`. `cl` is dimensionless and `dl` is
$\ell(\ell+1)C_\ell/2\pi$ in $\mu\mathrm{K}^2$. Pass `lensing_potential_ells`
to get the lensing potential as a fourth result, and `lensed_spectra` lenses
the spectra.

## Where to go next?

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} How it works
:link: pipeline
:link-type: doc

The four steps from the Einstein-Boltzmann solve to the lensed spectra.
:::

:::{grid-item-card} API reference
:link: api/reference
:link-type: doc

Every module, function and class.
:::

::::

```{toctree}
:hidden:

Quickstart <self>
pipeline
development
acknowledgement
```

```{toctree}
:hidden:
:caption: API Reference

api/reference
```
