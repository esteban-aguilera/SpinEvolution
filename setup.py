"""Run this file as:
python setup.py build_ext --inplace
"""

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy


# setup(
#     ext_modules=cythonize('spin_evolution/utils/*.pxd', annotate=True)
# )

setup(
    ext_modules=cythonize('spin_evolution/SpinLattice.pyx', annotate=True,
                          build_dir='cython'),
    include_dirs=[numpy.get_include()]
)
