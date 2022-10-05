# Run 'python setup.py build_ext --inplace' to setup cython

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy

extensions = [
  Extension("cython_col2im", ["cython_col2im.pyx"],
            include_dirs = [numpy.get_include()]
  ),
]

setup(
    ext_modules = cythonize(extensions),
)



