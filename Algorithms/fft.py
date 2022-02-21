import math

def FFT(A, w):
    n = len(A)
    assert w**n == 1
    assert all(w**k != 1 for k in range(1, n))
    #check that n is a power of 2
    assert math.log(n, 2).is_integer()
    if n == 1:
        return {1: A[0]} #(f(1) = a0)

    Fe = FFT(A[::2], w**2)
    Fo = FFT(A[1::2], w**2)
    F = {}
    
    for k in range(0, n):
        F[w**k] = Fe[w**(2*k)] + (w**k)*Fo[w**(2*k)]
    
    return F

