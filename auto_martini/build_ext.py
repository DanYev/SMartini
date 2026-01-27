from __future__ import annotations

"""UNMODIFIED ORIGINAL (kept as a literal block for reference)

from setuptools import setup, Extension, find_packages
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "auto_martiniM3.optimization_cy",
        ["auto_martiniM3/optimization_cy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-fopenmp", "-ffast-math", "-ftree-vectorize",
                           "-march=native", "-fopt-info-vec-optimized"],
        extra_link_args=["-fopenmp"]
    ),
]

setup(
    name="auto_martiniM3",
    version="0.0.1",
    packages=find_packages(),
    ext_modules=cythonize(extensions),
)

# python build_ext.py build_ext --inplace
"""

"""Poetry build hook for compiling Cython extensions.

Poetry (via poetry-core) executes this file as a script during wheel/editable
builds. It expects a function named `build(setup_kwargs)`.

When Poetry runs it, it does *not* pass setuptools-style command line args, so
this file must *not* behave like a `setup.py` script.
"""

import numpy
from Cython.Build import cythonize


def build(setup_kwargs: dict) -> None:
    """Populate setuptools kwargs with our extension modules (Poetry hook)."""

    setup_kwargs.update(
        ext_modules=cythonize(
            [
                "auto_martiniM3/optimization_cy.pyx",
            ],
            compiler_directives={"language_level": "3"},
        ),
        include_dirs=numpy.get_include(),
    )


if __name__ == "__main__":
    # poetry-core runs the build script as `python build_ext.py` during builds.
    # The actual hook entrypoint is `build(setup_kwargs)`, so when executed as a
    # script we must *not* error.
    pass