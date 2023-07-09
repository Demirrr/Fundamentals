"""
# Sorting Algorithms  Worst-case Runtime
# Insertion            O(n^2)
# Merge                O(n log(n)
# Heap                 O(n log(n))
# Quick                O(n^2)

All the aforementioned algorithms share an interesting property:
the sorted order they determine is based only on comparisons between the input elements.
We call such sorting algorithms comparison sorts.


# Counting             O(k+n)
# Radix                O(d(n+k))
# Bucket               O(n^2)

"""
import time
from typing import Iterable


def insertion_sort(arr: Iterable[object], ascending=True) -> Iterable[object]:
    """
    An incremental algorithm to sort a sequence of objects.

    arr: a sequence of (n) objects

    Complexity: O (n^2)

    Best Runtime Complexity : O(n)
    Worst-case Runtime Complexity: O(n^2)

    [5, 2, 4, 7, 1, 3, 2, 6]
    i=1, key=2, j=0:
            => insert the element located in the j.th position (2) into the j+1.position;
            => decrement j
            => reinsert the key into the j+1. position.
    [2, 5, 4, 7, 1, 3, 2, 6]
    i=2, key=4, j=1:
            => insert the element located in the j.th position (5) into the j+1.position;
            => decrement j
            => reinsert the key into the j+1. position.
    [2, 4, 5, 7, 1, 3, 2, 6]
    i=3, key=7, j=2:
            don't do anything
    [2, 4, 5, 7, 1, 3, 2, 6]
    i=4, key=1, j=3:
    => insert the element located in the j.th position (7) into the j+1.position;
    => decrement j
    => j=2

    => insert the element located in the j.th position (5) into the j+1.position;
    => decrement j
    => j=1

    => insert the element located in the j.th position (4) into the j+1.position;
    => decrement j
    => j=0

    => insert the element located in the j.th position (2) into the j+1.position;
    => decrement j
    => j=-1

    => reinsert the key into the j+1. position.
    [1, 2, 4, 5, 7, 3, 2, 6]
    ...
    """
    n = len(arr)
    # O(n): Iterate over n-1 elements.
    for i in range(1, n):
        # (1) Store the i.th element.
        key = arr[i]
        # (2) Store the left neighbour of the i.th element.
        j = i - 1
        # (3) O(n): Iterate over elements that are on the left side of the i.th element
        # starting from i-1 until we reached the first element in the arr or the value condition does not hold.
        while j >= 0 and ((arr[j] > key) == ascending):
            # (3.1) Insert the value denoted by the pointer into the subsequent right of the pointer.
            arr[j + 1] = arr[j]
            # (3.2) Reduce the pointer.
            j -= 1
        # (4) Insert the key value.
        arr[j + 1] = key
    return arr


def merge_sort(arr: Iterable[object]):
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

    if len(arr) < 2:
        return arr[:]
    else:
        middle = len(arr) // 2
        # (1) Sort the first half
        first_half = merge_sort(arr[:middle])
        # (2) Sort the second half
        second_half = merge_sort(arr[middle:])
        # (3) Merge (1) and (2) and return it
        return merge(first_half, second_half)


def bubble_sort(arr: Iterable[object]):
    """
    Complexity O(n^2)
    """
    swap = False
    # O(len(L))
    while not swap:
        swap = True
        # O(len(L))
        for j in range(1, len(arr)):
            if arr[j - 1] > arr[j]:
                swap = False
                temp = arr[j]
                arr[j] = arr[j - 1]
                arr[j - 1] = temp

    return arr


def selection_sort(A: Iterable[object]):
    """
    Complexity O(n^2)
    """
    suffixSt = 0
    # O(len(L))
    while suffixSt != len(A):
        # O(len(L))
        for i in range(suffixSt, len(A)):
            if A[i] < A[suffixSt]:
                A[suffixSt], A[i] = A[i], A[suffixSt]
        suffixSt += 1
    return A


def heap_sort(arr: Iterable[int]):
    """ Heap Sort Algorithm
    TLDR; Like insertion sort, but unlike merge sort. Inplace sort.

    Heap: ordered binary tree

    1. Build max heap from unordered array
    2. Find max element A[0]
    3. Swap elements A[n] with A[1]. Now max element at the end of the array
    4. Forget the last element A[n] ny decrementing the heap size
    5. New root of a subtree may violate max heap but its children do not violate. Go to (2)

    Visual Explanation:
    arr=[16, 14, 10, 8, 7, 9, 3, 2, 4, 1]
    arr[0] is the root of the tree.
                   (16)
             (4)          (10)
        (14)     (7)    (9)  (3)
      (7)  (8)  (1)

    Heap representation of an arr
            (0)
        (1)      (2)
      (3) (4)   (5) (6)
    (7)
    where (i) denotes the i.th element.
    parent(i) = (i-1)/2, left(i)   = 2(i+1) right(i)  = 2(i+2)

    Max-Heap Property: the key/ of a node parent >= child, e.g.  nearly complete binary tree
            (9)
        (8)      (3)
      (1) (5)   (2)

    O(n log(n))
    only a constant number of array elements are stored outside the input array at any time.
    (0)

                   (16)
             (4)          (10)
        (14)     (7)    (9)  (3)
      (7)  (8)  (1)

    (1)
                   (16)
             (14)          (10)
        (4)     (7)    (9)  (3)
      (7)  (8)  (1)

    (2)
                (16)
             (14)          (10)
        (8)     (7)    (9)  (3)
      (7)  (4)  (1)


    """

    def left_child(i):
        return 2 * i + 1

    def right_child(i):
        return 2 * i + 2

    def max_heapify(input_arr, heap_size: int, idx: int):
        """
        Correct a single violation of the heap property in the root of a subtree.
        heap_size: represents how many elements in the heap are stored within array.

        idx denotes the root node of a subtree
        """

        assert heap_size > idx
        # (1) Assign the largest
        largest = idx
        # (2) Get the two children of the largest
        idx_left_child, idx_right_child = left_child(largest), right_child(largest)
        # (3) Is left child greater ?
        if heap_size > idx_left_child and input_arr[idx_left_child] > input_arr[idx]:
            largest = idx_left_child
        # (3) Is right child greater ?
        if heap_size > idx_right_child and input_arr[idx_right_child] > input_arr[largest]:
            largest = idx_right_child
        # (4) is largest changed?
        if largest != idx:
            input_arr[idx], input_arr[largest] = input_arr[largest], input_arr[idx]
            max_heapify(input_arr, heap_size, largest)

    def build_max_heap(input_array):
        """
                    [0,  1,   2, 3, 4, 5, 6, 7, 8, 9]
        input_array=[16, 14, 10, 8, 7, 9, 3, 2, 4, 1]

        n=10, then iterate over idx=[4,3,2,1,0].
        """
        for _ in range(len(input_array) // 2 - 1, -1, -1):
            max_heapify(input_array, heap_size=len(input_array), idx=_)
    # (1) Build max heap from unordered array
    build_max_heap(arr)
    # Iterate from n-1 to 0 indexes
    for n in range(len(arr) - 1, 0, -1):
        # Exchange swap
        # (2) Find max element A[0]
        # (3) Swap elements A[n] with A[1]. Now max element at the end of the array
        # (4) Forget the last element A[n] ny decrementing the heap size
        arr[0], arr[n] = arr[n], arr[0]
        # (5) New root of a subtree may violate max heap but its children do not violate. Go to (2)
        max_heapify(arr, heap_size=n, idx=0)
    return arr

def quick_sort(arr: Iterable[int]):
    """
    The quicksort algorithm has a worst-case running time of O(n^2) on an input array of n numbers.
    Despite this slow worst-case running time, quicksort is often the best practical choice for sorting because it is remarkably efﬁcient on the average:
    its expected running time is O(n log(n)) and the constant factors hidden in the O(n log(n)) notation are quite small.
    It also has the advantage of sorting in place (see page 17),

    """

    def partition(array, low, high):
        # (1) Choose the last element as the pivot
        pivot = array[high]
        i = low - 1
        # (2) Iterate from low to high
        for j in range(low, high):
            # (3)
            if array[j] <= pivot:
                i += 1
                # Exchange.
                array[i], array[j] = array[j], array[i]
        # Exchange.
        array[i + 1], array[high] = array[high], array[i + 1]
        return i + 1

    def qs(array, low: int, high: int):
        if low < high:
            pivot_index = partition(array, low, high)

            qs(array, low, pivot_index - 1)
            qs(array, pivot_index + 1, high)

    qs(arr, low=0, high=len(arr) - 1)
    return arr


print(quick_sort([5, 2, 4, 7, 1, 3, 2, 6]))
print(heap_sort([5, 2, 4, 7, 1, 3, 2, 6]))
print(bubble_sort([5, 2, 4, 7, 1, 3, 2, 6]))
print(selection_sort([5, 2, 4, 7, 1, 3, 2, 6]))
print(insertion_sort([5, 2, 4, 7, 1, 3, 2, 6]))
