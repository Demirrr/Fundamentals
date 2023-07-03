"""
Understanding iterative and recursive computations.

# (1) Introduction
We show two different implementation of summation operation/computation.
By iterative_sum() and recursive_sum(), we would like to show that


# (2) Brute vs Divide-and-Conquer

In the example problem, our goal is to write an algorithm that buys at the lowest and sells at the highest.
"""
import random
from typing import Iterable


# (1) Introduction
def iterative_sum(arr):
    result = 0
    for i in arr:
        result += i
    return result
def recursive_sum(arr):
    if len(arr) == 1:
        return arr[0]
    elif len(arr) == 2:
        return arr[0] + arr[1]
    else:
        mid = len(arr) // 2
        return recursive_sum(arr[:mid]) + recursive_sum(arr[mid:])


for _ in range(10):
    x = [random.randint(0, 100) for _ in range(100)]
    assert iterative_sum(x) == sum(x) == recursive_sum(x)


# (2) Brute vs Divide-and-Conquer
def find_maximum_subarray_brute_force(arr: Iterable[object]):
    """ Compute all possible buy and sell days:O(n^2)."""
    pair = []
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            pair.append((arr[j] - arr[i], i, j))
    profit, i, j = sorted(pair, key=lambda x: x[0], reverse=True)[0]
    return profit


def find_maximum_subarray_recursive(arr):
    """
    O(n log(n)))

    T(n) = O(1) + 2T(n/2) + O(n) + O(1)
    =2T(n/2) + O(n)

    if n=1, T(n) = O(1)
    n>1, 2T(n/2) + O(n)

    Asymptotically faster than the brute-force method
    """
    counter=[]

    def fma(A, low, high):
        if low == high:
            return low, high, A[low]  # base case
        # (1) Find the middle index.
        mid = (low + high) // 2
        # (2) Solve the left. # O(n/2)
        left_low, left_high, left_sum = fma(A, low, mid)
        # (3) Solve the right.# O(n/2)
        right_low, right_high, right_sum = fma(A, mid + 1, high)
        # (4) Solve the crossing. O(n)
        cross_low, cross_high, cross_sum = fma_crossing(A, low, mid, high)
        counter.append((low,high))
        if left_sum >= right_sum and left_sum>= cross_sum:
            return left_low, left_high, left_sum
        elif right_sum >= left_sum and right_sum>= cross_sum:
            return right_low, right_high, right_sum
        else:
            return cross_low, cross_high, cross_sum

    def fma_crossing(arr, low, mid, high):
        left_sum = float('-inf')
        _sum = 0
        max_left = mid

        # Iterate from mid to low
        for i in range(mid, low - 1, -1):
            _sum += arr[i]
            if _sum > left_sum:
                left_sum = _sum
                max_left = i

        right_sum = float('-inf')
        _sum = 0
        max_right = mid + 1
        # Iterative from mid to high.
        for j in range(mid + 1, high + 1):
            _sum += arr[j]
            if _sum > right_sum:
                right_sum = _sum
                max_right = j

        return max_left, max_right, left_sum + right_sum
    # (1) Apply transformation
    # Store prices changes between each consecutive days.
    arr=[arr[i] - arr[i-1] for i in range(1,len(arr))]
    l, h, maximum_val = fma(arr, 0, len(arr) - 1)
    return sum(arr[l:h+1])

for _ in range(10):
    price_of_a_stock = [random.randint(10, 100) for _ in range(100)]
    assert find_maximum_subarray_brute_force(price_of_a_stock) == find_maximum_subarray_recursive(price_of_a_stock)