def square_matrix_mul(A, B):
    """
    Complexity O(n^3)
    """
    n = len(A)
    C = [[0 for __ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]
    return C


def strassen_square_matrix_mul(A, B):
    """
    O(n ^log(7))
    """
    n = len(A)
    C = [[0 for __ in range(n)] for _ in range(n)]

    if n == 1:
        return A[0] * B[0]
    else:
        C[0][0] = strassen_square_matrix_mul([A[0][0]], [B[0][0]]) + strassen_square_matrix_mul([A[0][1]], [B[1][0]])

        C[0][1] = strassen_square_matrix_mul([A[0][0]], [B[0][1]]) + strassen_square_matrix_mul([A[0][1]], [B[1][1]])

        C[1][0] = strassen_square_matrix_mul([A[1][0]], [B[0][0]]) + strassen_square_matrix_mul([A[1][1]], [B[1][1]])
        C[1][1] = strassen_square_matrix_mul([A[1][0]], [B[0][1]]) + strassen_square_matrix_mul([A[1][1]], [B[1][1]])

    return C
print(strassen_square_matrix_mul(A=[[0, 2],
                                     [2, 0]],
                                  B=[[1, 0],
                                     [0, 1]]))

print(square_matrix_mul(A=[[0, 2],
                           [2, 0]],
                        B=[[1, 0],
                           [0, 1]]))
