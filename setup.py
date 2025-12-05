from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "minimax_cy",
        ["minimax_cy.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3", "-march=native"],  # Aggressive optimization
    )
]

setup(
    name="chess_engine_cy",
    ext_modules=cythonize(extensions, language_level=3),
)
