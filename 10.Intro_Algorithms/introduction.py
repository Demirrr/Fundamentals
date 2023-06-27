from typing import Iterable


def insertion_sort(A: Iterable[int], ascending=False):
    """
    A: a sequence of (n) numbers

    """
    # O(n): Iterative over A
    for i in range(1, len(A)):
        # (1) Store the value pointed by i.
        key = A[i]
        j = i - 1
        # (2) O(n): Iterate over the right of side of A starting
        while j >= 0 and ((A[j] > key) == ascending):
            A[j + 1] = A[j]
            j -= 1
        A[j + 1] = key
    return A


## Insertion Sort
print(insertion_sort(A=[1, 3, 13, -2]))
