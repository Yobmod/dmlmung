from setuptools import setup
from Cython.Build import cythonize

# python compile.py build_ext --inplace
# cython myfile.pyx --embed
# gcc -Os -I /usr/include/python3.6 example.c -lpython3.6 -o example

setup(
    name='Hello world app',
    ext_modules=cythonize("fit_pal.pyx",
                          language_level="3",
                          annotate=True),
    # zip_safe=False,
)
