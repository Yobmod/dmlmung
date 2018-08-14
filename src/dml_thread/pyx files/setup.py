from distutils.core import setup
from Cython.Build import cythonize


scripts = [
            "somecython.pyx",
            "mung.pyx", 
            "plot.pyx", 
            # "gui.pyx",
            "log.pyx",
            "types.pyx",
]


setup(
    ext_modules=cythonize(scripts, annotate=True),
)
