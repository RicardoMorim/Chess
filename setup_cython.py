"""
Setup script for compiling Cython extensions.
Usage: python setup_cython.py build_ext --inplace
"""

from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "minimax_improved_cy",
        ["minimax_improved_cy.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"],  # Maximum optimization
    )
]

setup(
    name="minimax_improved_cy",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': 3,
            'boundscheck': False,
            'wraparound': False,
            'cdivision': True,
        }
    ),
)
