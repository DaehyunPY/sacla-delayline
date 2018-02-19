from distutils.core import setup

from Cython.Build import cythonize
from numpy import get_include

setup(
    name='saclatools',
    version='20170220',
    author='Daehyun You',
    author_email='daehyun@dc.tohoku.ac.jp',
    packages=['saclatools'],
    ext_modules=cythonize("saclatools/lma_fmt.pyx"),
    include_dirs=[get_include()]
)
