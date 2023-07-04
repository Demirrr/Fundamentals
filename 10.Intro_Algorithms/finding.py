from typing import Iterable
import random


# Peak Finding in 1D.# an element  is a peak in 1D if if its greater or equal than all its two neighbors
def peak_finding1D():
    def iterative_peak_finding_1d(array) -> int:
        """ Find the first peak """
        # First and last item cannot be peak
        for i in range(1, len(array) - 1):
            # Not in the cor
            if array[i] >= array[i - 1] and array[i] >= array[i + 1]:
                return i
        return None

    def recursive_peak_finding_1d(array, left_idx, right_idx) -> int:
        """ Find a peak in 1D."""
        # (1) Find the middle index between left and right indexes.
        mid_idx = (right_idx - left_idx) // 2
        # (2) Store the value of middle index btw left and right indexes.
        mid_val = array[mid_idx]
        # (3) Store the value of left and right indexes.
        left_mid_val = array[mid_idx - 1]
        # (4) Store the values of left and right indexes.
        right_mid_val = array[mid_idx + 1]

        # (5) Check if (2) is a peak value.
        if mid_val >= left_mid_val and mid_val >= right_mid_val:
            return mid_idx

        # (7) Check if (3) is greater than (2).
        if mid_val < left_mid_val:
            return recursive_peak_finding_1d(array, mid_idx, right_idx - 1)

        # (6) Check if (4) is greater than (2).
        if mid_val < right_mid_val:
            return recursive_peak_finding_1d(array, left_idx + 1, mid_idx)

    # Sanity checking
    for _ in range(10):
        numbers = [random.randint(0, 10) for i in range(10)]
        idx_peak = iterative_peak_finding_1d(numbers)
        assert numbers[idx_peak - 1] <= numbers[idx_peak] >= numbers[idx_peak + 1]
        idx_peak = recursive_peak_finding_1d(numbers, 0, len(numbers))
        assert numbers[idx_peak - 1] <= numbers[idx_peak] >= numbers[idx_peak + 1]


def peak_finding2D():
    # Peakfinding2D
    grid = [[0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0]]
    # Peak cannot appear near borders
    grid[random.randint(1, len(grid) - 2)][random.randint(1, len(grid[0]) - 2)] = 1

    def iterative_peak_finding_2d(g):
        # to be a peak
        # (1) Not being in the border
        # (2) Being larger than any of the 4 or 8 neighbour
        m, n = len(g), len(g[0])
        for i in range(1, m - 1):
            for j in range(1, n - 1):
                if g[i][j] > g[i][j + 1] and g[i][j] > g[i][j - 1] and g[i][j] > g[i + 1][j] and g[i][j] > g[i - 1][j]:
                    return i, j
        return False

    print(grid)
    print(f'Iterative 2D Peak Finder => Coordinate of the peak:{iterative_peak_finding_2d(grid)}')

def finding_maximum_subarray():
    # Peak Finding in 2D.
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
    assert find_maximum_subarray_brute_force(A=arr) == find_maximum_subarray_recursive(arr)

peak_finding1D()
peak_finding2D()
finding_maximum_subarray()
