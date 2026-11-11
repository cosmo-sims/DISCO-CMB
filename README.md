<h1 align="center">DISCO-CMB</h1>

<!--
The coverage badge reads a gist that the `tests` workflow rewrites on every
push to main; the id below is the repository variable COVERAGE_GIST_ID.
These badges sit above the `overview` snippet marker, so the documentation
site does not repeat them.
-->
<p align="center">
  <a href="https://github.com/cosmo-sims/DISCO-CMB/actions/workflows/tests.yml"><img alt="tests" src="https://github.com/cosmo-sims/DISCO-CMB/actions/workflows/tests.yml/badge.svg?branch=main"></a>
  <a href="https://github.com/cosmo-sims/DISCO-CMB/actions/workflows/tests.yml"><img alt="coverage" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fgist.githubusercontent.com%2Flawrenceberry%2Fb451e14a9620fabb0d49c345f13b5414%2Fraw%2Fdiscocmb-coverage.json"></a>
  <a href="https://github.com/cosmo-sims/DISCO-CMB/actions/workflows/lint.yml"><img alt="lint" src="https://github.com/cosmo-sims/DISCO-CMB/actions/workflows/lint.yml/badge.svg?branch=main"></a>
  <a href="https://github.com/cosmo-sims/DISCO-CMB/actions/workflows/types.yml"><img alt="types" src="https://github.com/cosmo-sims/DISCO-CMB/actions/workflows/types.yml/badge.svg?branch=main"></a>
  <a href="https://pypi.org/project/discocmb/"><img alt="PyPI" src="https://img.shields.io/pypi/v/discocmb"></a>
  <a href="https://pypi.org/project/discocmb/"><img alt="Downloads" src="https://img.shields.io/pypi/dm/discocmb"></a>
  <a href="https://pypi.org/project/discocmb/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/discocmb"></a>
  <a href="https://docs.jax.dev/"><img alt="JAX" src="https://img.shields.io/badge/JAX-compatible-3f51b5"></a>
  <a href="https://arxiv.org/abs/2311.03291"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2311.03291-b31b1b.svg"></a>
  <a href="https://www.conventionalcommits.org/en/v1.0.0/"><img alt="Conventional Commits" src="https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white"></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
</p>

<p align="center">
  <a href="#installation">Installation</a> •
  <a href="#quickstart">Quickstart</a> •
  <a href="https://cosmo-sims.github.io/DISCO-CMB/">Documentation</a> •
  <a href="https://arxiv.org/abs/2311.03291">DISCO-EB paper</a> •
  <a href="#citation">Citation</a>
</p>

<!-- start:overview -->

DISCO-CMB computes CMB angular power spectra for **batches of cosmologies on the GPU**. It builds on the Einstein-Boltzmann solver of [DISCO-EB](https://github.com/cosmo-sims/DISCO-EB) and adds line-of-sight integration, the unlensed TT, TE and EE spectra, the lensing potential and the lensed TT, TE, EE and BB spectra.

A single GPU launch evolves every wave mode of every cosmology in a batch, with the line-of-sight sources evaluated inside the solver's kernel. The unlensed TT spectrum agrees with CLASS to 0.25 percent for multipoles 2 to 1200, and JAX can differentiate the spectra with respect to the cosmological parameters.

<!-- end:overview -->

The [documentation](https://cosmo-sims.github.io/DISCO-CMB/) covers the API
and explains how the pipeline fits together.

## Installation

<!-- start:install -->

DISCO-CMB is not on PyPI yet. It needs an NVIDIA GPU and a checkout of DISCO-EB beside it, which [uv](https://docs.astral.sh/uv/) installs as a path dependency:

```bash
git clone --branch v2 https://github.com/cosmo-sims/DISCO-EB.git
git clone https://github.com/cosmo-sims/DISCO-CMB.git
cd DISCO-CMB && uv sync
```

<!-- end:install -->

## Quickstart

<!-- start:quickstart -->

This example computes the unlensed spectra of three cosmologies that differ in their spectral index. The grids come from a fiducial cosmology once, and every later batch reuses them:

```python
from discocmb import cmb_grids, compute_cl_power_spectrum_batch

planck = {
    "Omegam": 0.3144,
    "Omegab": 0.04939,
    "H0": 67.32,
    "Omegak": 0.0,
    "A_s": 2.1e-9,
    "n_s": 0.966,
    "k_p": 0.05,
    "Tcmb": 2.7255,
    "YHe": 0.2454,
    "Neff": 3.046,
    "Nmnu": 0.0,
    "mnu": 0.0,
    "w_DE_0": -1.0,
    "w_DE_a": 0.0,
    "cs2_DE": 1.0,
}
batch = [planck | {"n_s": n_s} for n_s in (0.95, 0.966, 0.98)]

grids = cmb_grids(planck)
ells, cl, dl = compute_cl_power_spectrum_batch(batch, grids=grids)
dl.tt  # (3, n_ell) in muK^2
```

<!-- end:quickstart -->

## Development

<!-- start:development -->

The repository uses [uv](https://docs.astral.sh/uv/). After the installation above:

```bash
uv run pytest tests         # needs a CUDA GPU
uv run ruff format && uv run ruff check --fix
uv run ty check
uv run pre-commit install   # the checks above, and a commit-message check
```

The documentation (Sphinx with MyST and autodoc) builds in an environment of its own, the package plus the docs group:

```bash
export UV_PROJECT_ENVIRONMENT=.venv-docs
uv sync --no-default-groups --group docs
uv run --no-sync sphinx-autobuild docs docs/_build/html   # live preview
uv run --no-sync sphinx-build -W docs docs/_build/html    # what CI runs
```

<!-- end:development -->

## Citation

<!-- start:citation -->

If you use DISCO-CMB in your research, please cite this repository (see
[CITATION.cff](https://github.com/cosmo-sims/DISCO-CMB/blob/main/CITATION.cff))
and the DISCO-EB article it builds on:

```bibtex
@article{DISCO-EB:2024,
    author = "Hahn, Oliver and List, Florian and Porqueres, Natalia",
    title = "{DISCO-DJ I: a differentiable Einstein-Boltzmann solver for cosmology}",
    eprint = "2311.03291",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    doi = "10.1088/1475-7516/2024/06/063",
    journal = "JCAP",
    volume = "06",
    pages = "063",
    year = "2024"
}
```

<!-- end:citation -->

## License

<!-- start:license -->

DISCO-CMB is released under the
[MIT License](https://github.com/cosmo-sims/DISCO-CMB/blob/main/LICENSE).

<!-- end:license -->
