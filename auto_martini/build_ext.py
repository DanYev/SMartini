r"""Poetry build hook for compiling Cython extensions.

Poetry (via poetry-core) executes this file as a script during wheel/editable
builds. It expects a function named `build(setup_kwargs)`.

Do NOT call setuptools.setup() at import time here.
"""

import numpy
from Cython.Build import cythonize


def build(setup_kwargs):
    """Populate setuptools kwargs with our extension modules."""
    setup_kwargs.update(
        ext_modules=cythonize(
            [
                "auto_martiniM3/optimization_cy.pyx",
                "auto_martiniM3/energy_cy.pyx",
            ],
            compiler_directives={"language_level": "3"},
        ),
        include_dirs=numpy.get_include(),
    )
    
# from setuptools import setup
# from Cython.Build import cythonize
# import numpy

# setup(
#     ext_modules=cythonize(
#         [
#             "auto_martiniM3/optimization_cy.pyx",
#             "auto_martiniM3/energy_cy.pyx",
#         ],
#         compiler_directives={"language_level": "3"},
#         # extra_compile_args=["-O3", "-fopenmp", "-ffast-math", "-ftree-vectorize",
#         #                    "-march=native", "-fopt-info-vec-optimized"],  
#         # extra_link_args=["-fopenmp"]
#     ),
#     include_dirs=numpy.get_include(),
# )
