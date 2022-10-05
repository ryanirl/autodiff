# Use 'python3 setup.py build_ext --inplace' to compile.

from setuptools import setup, Extension


setup(
    ext_modules = [Extension("_col2im", ["col2im.c"])]
)





