from setuptools import Extension, setup

import numpy as np
from Cython.Build import cythonize

extensions = [
    Extension(
        name="ligpar_cy",
        sources=["ligpar_cy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],
    )
]

setup(
    name="ligpar-cy",
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    ),
)
