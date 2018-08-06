from distutils.core import setup
from Cython.Build import cythonize


scripts = [
            "somecython.pyx",
            # "mung.pyx", 
            # "plot.pyx", 
            # "gui.pyx",
]


setup(
    ext_modules=cythonize(scripts, annotate=True),
)
