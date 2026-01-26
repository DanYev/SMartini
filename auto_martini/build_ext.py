from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        [
            "auto_martiniM3/optimization_cy.pyx",
            "auto_martiniM3/energy_cy.pyx",
        ],
        compiler_directives={"language_level": "3"},
        # extra_compile_args=["-O3", "-fopenmp", "-ffast-math", "-ftree-vectorize",
        #                    "-march=native", "-fopt-info-vec-optimized"],  
        # extra_link_args=["-fopenmp"]
    ),
    include_dirs=numpy.get_include(),
)
