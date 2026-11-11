import os

# The perturbation solver is a numba-CUDA kernel launched beside JAX, whose
# per-thread local memory the driver reserves outside JAX's pool; let the pool
# grow on demand rather than preallocating 75 percent of the card.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax

jax.config.update("jax_enable_x64", True)


def pytest_sessionstart(session):
    jax.print_environment_info()
