import cython


#@cython.ccall
def somemath(number: cython.int) -> cython.int:
    dn: cython.int = number * 6
    return dn