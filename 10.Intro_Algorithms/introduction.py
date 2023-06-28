from typing import Iterable

def bubble_sort(A: Iterable[object]):
    """
    Complexity O(n^2)
    """
    swap = False
    # O(len(L))
    while not swap:
        swap = True
        # O(len(L))
        for j in range(1, len(A)):
            if A[j - 1] > A[j]:
                swap = False
                temp = A[j]
                A[j] = A[j - 1]
                A[j - 1] = temp

    return A


def insertion_sort(A: Iterable[int], ascending=True):
    """
    A: a sequence of (n) numbers

    Complexity: O (n^2)

    Best Runtime Complexity : O(n)
    Worst-case Runtime Complexity: O(n^2)

    """
    # O(n): Iterative over A.
    for i in range(1, len(A)):
        # (1) Store the key value pointed by the i.th index.
        key = A[i]
        # (2) Store the left neighbour of the i.th index as pointer.
        j = i - 1
        # (3) O(n): Iterate over A, e.g., A[j], A[j-1], A[j-2] s.t. j>= 0 and the value condition holds.
        while j >= 0 and ((A[j] > key) == ascending):
            # (3.1) Insert the value denoted by the pointer into the subsequent right of the pointer.
            A[j + 1] = A[j]
            # (3.2) Reduce the pointer.
            j -= 1
        # (4) Insert the key value.
        A[j + 1] = key
    return A


def merge_sort(A: Iterable[object]):
    """
    Complexity: O (n log(n)

    A=[5, 2, 4, 7, 1, 3, 2, 6]

    Divide 1 => [5, 2, 4, 7]     [1, 3, 2, 6]

    Divide 2 => [5, 2] [4, 7]    [1, 3]  [2, 6]

    Concur 1 => [2, 5] [4, 7]    [1, 3]  [2, 6]

    Concur 2 => [2, 4, 5, 7]     [1, 2, 3, 6]

    Concur 3 => [1,2,3,4,5,6,7]
    """
    result = []

    def merge(left, right):
        # (1) Initialize pointers.
        i, j = 0, 0
        while i < len(left) and j < len(right):
            # (2) If pointer i  points a lower object, than
            # (2.1) add this object into result and increment i.
            if left[i] < right[j]:
                result.append(left[i])
                i += 1
            else:
                # (3) If the pointer i does not point a lower object, than
                # (3.1) Add the object pointed by j and increment j.
                result.append(right[j])
                j += 1
        if i < len(left):
            # Sanity checking
            assert (j < len(right)) == False
            # Add the resulting elements
            while i < len(left):
                result.append(left[i])
                i += 1

        if j < len(right):
            # Sanity checking
            assert (i < len(left)) == False
            # Add the resulting elements
            while j < len(right):
                result.append(right[j])
                j += 1
        return result

    if len(A) < 2:
        return A[:]
    else:
        middle = len(A) // 2
        # (1) Sort the first half
        first_half = merge_sort(A[:middle])
        # (2) Sort the second half
        second_half = merge_sort(A[middle:])
        # (3) Merge (1) and (2) and return it
        return merge(first_half, second_half)


print(insertion_sort(A=[5, 2, 4, 7, 1, 3, 2, 6]))
print(merge_sort(A=[5, 2, 4, 7, 1, 3, 2, 6]))
print(bubble_sort(A=[5, 2, 4, 7, 1, 3, 2, 6]))
