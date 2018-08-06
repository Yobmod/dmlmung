from distutils.core import setup
from Cython.Build import cythonize


scripts = ["somecython.pyx", "cv.pyx"]


setup(
    ext_modules=cythonize(scripts, annotate=True),
)
