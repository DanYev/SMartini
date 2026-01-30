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
from setuptools import Extension, setup, find_packages


def build(setup_kwargs: dict) -> None:
    """Populate setuptools kwargs with our extension modules (Poetry hook)."""

    # Build with OpenMP if the compiler supports it (Linux gcc/clang).
    # If OpenMP isn't available, compilation may fail; in that case remove
    # "-fopenmp" or adjust for your compiler toolchain.
    ext = Extension(
        name="auto_martiniM3.optimization_cy",
        sources=["auto_martiniM3/optimization_cy.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-ffast-math", "-ftree-vectorize", "-fopenmp"],
        extra_link_args=["-fopenmp"],
    )

    setup_kwargs.update(
        ext_modules=cythonize(
            [ext],
            compiler_directives={"language_level": "3"},
        ),
    )


def _make_extensions():
    """Extension list shared by Poetry hook and manual setuptools builds."""
    return [
        Extension(
            name="auto_martiniM3.optimization_cy",
            sources=["auto_martiniM3/optimization_cy.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3", "-ffast-math", "-ftree-vectorize", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        )
    ]


if __name__ == "__main__":
    # Allow manual rebuilds without Poetry:
    #   python build_ext.py build_ext --inplace
    setup(
        name="auto_martiniM3",
        version="0.0.0",
        packages=find_packages(),
        ext_modules=cythonize(
            _make_extensions(),
            compiler_directives={"language_level": "3"},
        ),
    )