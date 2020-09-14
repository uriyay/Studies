import random

def radix_sort(arr, base, digits):
    # See https://en.wikipedia.org/wiki/Radix_sort
    for i in range(0, digits):
        arr = sorted(arr, key=lambda x: (x % (base**(i+1))) // (base**i))
    return arr
    
def test():
    base = 10
    digits = 4
    arr = [random.randint(base**0, base**digits) for x in range(1000)]
    sorted_arr = radix_sort(arr, digits + 1)
    assert sorted_arr == sorted(arr)
    
if __name__ == '__main__':
    test()
