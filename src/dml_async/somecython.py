import cython

# c_int = cython.int  # reveal_type(cython.int)
c_int = cython.typedef(cython.bint) 


@cython.ccall
def somemath(number: c_int) -> c_int:
  dn: c_int = number * 6
  return dn

