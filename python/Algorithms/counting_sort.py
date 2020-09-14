def counting_sort(A, k):
    # https://en.wikipedia.org/wiki/Counting_sort
    B = [0] * len(A)
    C = [0] * k
    for i in range(len(A)):
        C[A[i]] += 1
    # C[i] now contain the number of occurences of i in A
    for j in range(1, k):
        C[j] += C[j - 1]
    # C[i] now contain the number of elements that are less than or equal to i
    for l in range(len(A) - 1, -1, -1):
        B[C[A[l]] - 1] = A[l]
        C[A[l]] -= 1
    return B
    
def test():
    arr = [random.randint(base**0, base**digits) for x in range(1000)]
    sorted_arr = counting_sort(arr, max(arr))
    assert sorted_arr == sorted(arr)
    
if __name__ == '__main__':
    test()
