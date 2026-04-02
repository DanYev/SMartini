from __future__ import annotations
import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup, find_packages


def build(setup_kwargs: dict) -> None:
    """Populate setuptools kwargs with our extension modules (Poetry hook)."""

    # Build with OpenMP if the compiler supports it (Linux gcc/clang).
    # If OpenMP isn't available, compilation may fail; in that case remove
    # "-fopenmp" or adjust for your compiler toolchain.
    ext = [
        Extension(
            name="smartini.optimization_cy",
            sources=["smartini/optimization_cy.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3", "-ffast-math", "-ftree-vectorize", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        ),
        Extension(
            name="smartini.ligpar_cy",
            sources=["smartini/ligpar_cy.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3", "-ffast-math", "-ftree-vectorize", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        ),
    ]

    setup_kwargs.update(
        ext_modules=cythonize(
            ext,
            compiler_directives={"language_level": "3"},
        ),
    )


def _make_extensions():
    """Extension list shared by Poetry hook and manual setuptools builds."""
    return [
        Extension(
            name="smartini.optimization_cy",
            sources=["smartini/optimization_cy.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3", "-ffast-math", "-ftree-vectorize", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        ),
        Extension(
            name="smartini.ligpar_cy",
            sources=["smartini/ligpar_cy.pyx"],
            include_dirs=[numpy.get_include()],
            extra_compile_args=["-O3", "-ffast-math", "-ftree-vectorize", "-fopenmp"],
            extra_link_args=["-fopenmp"],
        ),
    ]


if __name__ == "__main__":
    # Allow manual rebuilds without Poetry:
    #   python build_ext.py build_ext --inplace
    import sys
    # If invoked without commands (e.g., by Poetry's build system), assume build_ext
    if len(sys.argv) == 1:
        sys.argv.extend(["build_ext", "--inplace"])
    
    setup(
        name="smartini",
        version="0.0.1",
        packages=find_packages(),
        ext_modules=cythonize(
            _make_extensions(),
            compiler_directives={"language_level": "3"},
        ),
    )