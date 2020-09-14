import random

def radix_sort(arr, d):
    for i in range(0, d):
        arr = sorted(arr, key=lambda x: (x % (10**(i+1))) // (10**i))
    return arr
    
def test():
    arr = [random.randint(10**0, 10**4) for x in range(1000)]
    sorted_arr = radix_sort(arr, 5)
    assert sorted_arr == sorted(arr)
    
if __name__ == '__main__':
    test()
