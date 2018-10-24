#cython: language_level=3
try:
    from setuptools import setup
    from setuptools import Extension
except ImportError:
    from distutils.core import setup
    from distutils.extension import Extension

from Cython.Build import cythonize
import numpy

scripts = [
            Extension("somecython", ["somecython.pyx"]),
            Extension("mung", ["mung.pyx"], include_dirs=[numpy.get_include()]),
            # "mung.pyx", 
            # "plot.pyx", 
            # "gui.pyx",
            #  "log.pyx",
            # "types.pyx",
]


setup(
    ext_modules=cythonize(scripts,
                          build_dir="build",  # puts .c and .html here
                          # annotate=True
    )
)
