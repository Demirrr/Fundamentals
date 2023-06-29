from typing import Iterable


def find_maximum_subarray_brute_force(A: Iterable[object]):
    """
    O(n^2)
    """
    pair = []
    for i, value_i in enumerate(A):
        for j, value_j in enumerate(A):
            if i == j:
                continue
            pair.append((sum(A[i:j]), i, j))

    res, i, j = sorted(pair, key=lambda x: x[0], reverse=True)[0]
    return sum(A[i:j])




def find_maximum_subarray_recursive(A):

    def fma(A, low, high):
        if low == high:
            return low, high, arr[low]  # base case
        # (1) Find the middle index.
        mid = (low + high) // 2
        # (2) Solve the left
        left_low, left_high, left_sum = fma(A, low, mid)
        # (3) Solve the right
        right_low, right_high, right_sum = fma(A, mid + 1, high)
        # (4) Solve the crossing
        cross_low, cross_high, cross_sum = fma_crossing(A, low, mid, high)

        if left_sum >= right_sum >= cross_sum:
            return left_low, left_high, left_sum
        elif right_sum >= left_sum >= cross_sum:
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

    low, high, maximum_val = fma(A, 0, len(A) - 1)
    return sum(A[low:high + 1])


arr = [13, -3, -25, 20, -3, -16, -23, 18, 20, -7, 12, -5, -22, 15, -4, 7]
print(find_maximum_subarray_brute_force(A=arr))
print(find_maximum_subarray_recursive(arr))
