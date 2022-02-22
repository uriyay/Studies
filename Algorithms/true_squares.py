from helper import one_based_range, OneBasedArray

def FLIP(x):
    return not x

def TRUE_SQUARES(arr):
    A = OneBasedArray(arr)
    n = len(A)
    for i in one_based_range(1, n):
        A[i] = False

    for i in one_based_range(1, n):
        k = i
        while (k <= n):
            A[k] = FLIP(A[k])
            k += i

    return A
