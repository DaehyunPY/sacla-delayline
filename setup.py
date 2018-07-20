from distutils.core import setup

from Cython.Build import cythonize
from numpy import get_include

setup(
    name='saclatools',
    version='20180720.0',
    author='Daehyun You',
    author_email='daehyun@dc.tohoku.ac.jp',
    packages=['saclatools'],
    ext_modules=cythonize("saclatools/lma_fmt.pyx"),
    include_dirs=[get_include()],
    install_requires=['cython', 'numpy', 'numba', 'cytoolz', 'pandas', 'pyspark'],
    extras_require={
        'hpc': ['dbpy', 'stpy'],
    },
)
