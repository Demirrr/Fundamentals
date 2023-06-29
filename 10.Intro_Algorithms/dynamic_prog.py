from typing import List


def lcs_length(X, Y):
    m = len(X)
    n = len(Y)

    c = [[0] * (n + 1) for _ in range(m + 1)]
    b = [[""] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        c[i][0] = 0

    for j in range(n + 1):
        c[0][j] = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                c[i][j] = c[i - 1][j - 1] + 1
                b[i][j] = "-"
            elif c[i - 1][j] >= c[i][j - 1]:
                c[i][j] = c[i - 1][j]
                b[i][j] = "↑"
            else:
                c[i][j] = c[i][j - 1]
                b[i][j] = "←"

    return c, b


# Test the code
X = "ABCBDAB"
Y = "BDCAB"

c, b = lcs_length(X, Y)

print("LCS Length Table:")
for row in c:
    print(row)

print("\nLCS Direction Table:")
for row in b:
    print(row)


exit(1)
def extended_bottom_up_cut_rod(p, n):
    r = [0] * (n + 1)
    s = [0] * (n + 1)

    for j in range(1, n + 1):
        q = float('-inf')
        for i in range(1, j + 1):
            if q < p[i] + r[j - i]:
                q = p[i] + r[j - i]
                s[j] = i
        r[j] = q

    return r, s


def print_cut_rod_solution(p, n):
    r, s = extended_bottom_up_cut_rod(p, n)
    while n > 0:
        print(s[n], end=" ")
        n -= s[n]
    print()


# Test the code
prices = [0, 1, 5, 8, 9, 10, 17, 17, 20]
rod_length = 8

max_value, cuts = extended_bottom_up_cut_rod(prices, rod_length)
print(f"The maximum obtainable value for rod length {rod_length} is: {max_value}")
print("Cutting scheme:", end=" ")
print_cut_rod_solution(prices, rod_length)



def bottom_up_cut_rod(p, n):
    r = [0] * (n + 1)

    for j in range(1, n + 1):
        q = float('-inf')
        for i in range(1, j + 1):
            q = max(q, p[i] + r[j - i])
        r[j] = q

    return r[n]


# Test the code
prices = [0, 1, 5, 8, 9, 10, 17, 17, 20]
rod_length = 5

max_value = bottom_up_cut_rod(prices, rod_length)
print(f"The maximum obtainable value for rod length {rod_length} is: {max_value}")



def memoized_cut_rod(p, n):
    r = [-1] * (n + 1)
    return memoized_cut_rod_aux(p, n, r)


def memoized_cut_rod_aux(p, n, r):
    if r[n] >= 0:
        return r[n]

    if n == 0:
        q = 0
    else:
        q = float('-inf')
        for i in range(1, n + 1):
            q = max(q, p[i] + memoized_cut_rod_aux(p, n - i, r))

    r[n] = q
    return q


# Test the code
prices = [0, 1, 5, 8, 9, 10, 17, 17, 20]
rod_length = 5

max_value = memoized_cut_rod(prices, rod_length)
print(f"The maximum obtainable value for rod length {rod_length} is: {max_value}")



def print_cut_rod_solution(p, n):
    r, s = extended_bottom_up_cut_rod(p, n)
    while n > 0:
        print(s[n])
        n -= s[n]


# Test the code
prices = [0, 1, 5, 8, 9, 10, 17, 17, 20]
rod_length = 8

print("Cutting scheme:")
print_cut_rod_solution(prices, rod_length)

exit(1)

##longest common subsequence
def lcs(X,Y):
    pass


# matrix-chain multiplication.
def matrix_chan_mul(A1, A2, A3, A4):
    # A1, A2, A3, A4

    # (1) (A1(A2(A3A4)))
    # (2) (A1 ( (A2A3) A4))
    # (3) ((A1A2)(A3A4))
    # (4) ( ( (A1 A2) A3) A4)

    def mm(A, B):

        m, n = len(A), len(A[0])
        p, q = len(B), len(B[0])
        assert n == q

        C = [[0 for __ in range(q)] for _ in range(m)]
        for i in range(m):
            for j in range(q):
                for k in range(p):
                    C[i][j] += A[i][k] * B[k][j]
        return C

    a1 = mm(A1, mm(A2, mm(A3, A4)))
    a2 = mm(A1, mm(mm(A2, A3), A4))
    a3 = mm(mm(A1, A2), mm(A3, A4))
    a4 = mm(mm(mm(A1, A2), A3), A4)

    print(a1)
    print(a2)
    print(a3)
    print(a4)


matrix_chan_mul(A1=[[1, 2], [1, 2]], A2=[[1, 2], [1, 2]], A3=[[1, 2], [1, 2]], A4=[[1, 2], [1, 2]])



def rod_cutting(lengths, prices, rod_length: int):
    """the rod cutting problem Given a rod of length n and prices

    For a given rod of length n and respective prices, the goal is to find
    a cut, cuts, or no cut that yields the maximum profit.

    There are 2^{n-1} cuts

    """

    n = len(lengths)
    # Create a table to store the maximum obtainable value for each rod length
    table = [0 for _ in range(rod_length + 1)]
    for i in range(1, rod_length + 1):
        max_value = float("-inf")
        for j in range(n):
            if lengths[j] <= i:
                max_value = max(max_value, prices[j] + table[i - lengths[j]])
        table[i] = max_value
    return table[rod_length]


def rod_cutting_recursive(lengths, prices, rod_length):
    n = len(lengths)

    def _rod_cutting_recursive(lengths, prices, n, rod_length):
        if n == 0 or rod_length == 0:
            return 0

        if lengths[n - 1] > rod_length:
            return _rod_cutting_recursive(lengths, prices, n - 1, rod_length)

        return max(prices[n - 1] + _rod_cutting_recursive(lengths, prices, n, rod_length - lengths[n - 1]),
                   _rod_cutting_recursive(lengths, prices, n - 1, rod_length))

    return _rod_cutting_recursive(lengths, prices, n, rod_length)


print(rod_cutting(lengths=[1, 2, 3, 4, 5, 6, 7, 8], prices=[1, 5, 8, 9, 10, 17, 17, 20], rod_length=4))

print(rod_cutting_recursive(lengths=[1, 2, 3, 4, 5, 6, 7, 8], prices=[1, 5, 8, 9, 10, 17, 17, 20], rod_length=4))
