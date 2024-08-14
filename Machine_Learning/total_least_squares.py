"""
# Total Least Squares
https://en.wikipedia.org/wiki/Total_least_squares

https://people.duke.edu/~hpgavin/SystemID/References/Markovsky+VanHuffel-SP-2007.pdf
https://scikit-learn.org/stable/modules/linear_model.html#ordinary-least-squares
https://people.duke.edu/~hpgavin/SystemID/CourseNotes/TotalLeastSquares.pdf

"""
import scipy
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Calculate Mean Squared Error for both models
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


# TLS implementation (assuming X and Y are column vectors) without bias term
def total_least_squares(X, Y):
    """
    Calculate the Total Least Squares (TLS) solution for linear regression.

    https://docs.scipy.org/doc/scipy/reference/odr.html#usage-information

    Parameters:
        X : array-like, shape (n_samples, n_features)
            The input data matrix.

        Y : array-like, shape (n_samples,)
            The target values vector.

    Returns:
        a_tls : array-like, shape (n_features,)
            The TLS coefficients (slopes) for the regression line.

        yhat : array-like, shape (n_samples,)
            Predicted target values using the TLS solution.

    Example usage:

    >>> X = np.array([[1, 1], [5, 25], [7, 49]])
    >>> Y = np.array([1, 3, 8]).reshape(len(X), -1)
    >>> a_tls, yhat = total_least_squares(X[:,0].reshape(-1,1), Y)
    >>> print(a_tls)
    [[-0.54921001]
     [ 0.23980658]]

    >>> print(yhat)
    [[0.03659332]
     [3.18328843]
     [7.93087704]]

    Source: https://people.duke.edu/~hpgavin/SystemID/CourseNotes/TotalLeastSquares.pdf
    """

    # Ensure X and Y are column vectors
    n, d = X.shape
    A = np.hstack([X, Y])
    # Compute SVD
    U, S, Vt = np.linalg.svd(A)
    # The last row of V corresponds to the singular vector associated with the smallest singular value
    V = Vt.T
    # Calculate TLS coefficients (slopes) using the last column of V and dividing by its last element
    weights = - V[:d, d:] / V[d:, d:]
    # Compute the projection of A onto the orthogonal complement of the singular vector associated with the smallest singular value
    Xtyt = - A.dot(V[:, d:]).dot(V[:, d:].T)
    # Calculate the error in X and Y using the projection matrix
    Xt = Xtyt[:, :d]
    # Predicted target values using TLS solution
    yhat = (X + Xt).dot(weights)
    return float(weights[0][0]), yhat

def ols(X, Y):
    model = LinearRegression(fit_intercept=False)
    model.fit(X, Y)
    return float(model.coef_[0][0]), model.predict(X)

# Set random seed for reproducibility
np.random.seed(42)

n = 150
noise_levels_X = np.linspace(0.1, 5, 4)  # Noise levels for X from 0 to 5 with 10 different values
noise_levels_Y = np.linspace(0.1, 5, 4)  # Noise levels for Y from 0 to 5 with 10 different values

is_tls_winner=[]
for x_error_rate in noise_levels_X:
    for y_error_rate in noise_levels_Y:
        # True slope is 2.0
        xerr = np.abs(np.random.normal(0, x_error_rate, n))
        X = np.random.normal(np.linspace(0, 10, n), xerr, n).reshape(n,-1)
        y = np.linspace(0, 20, n)
        yerr = np.abs(np.random.normal(0, y_error_rate, n))
        y = np.random.normal(y, yerr).reshape(n,-1)


        w_tls, yhat_tls =total_least_squares(X=X,Y=y)


        w_ols,yhat_ols=ols(X=X,Y=y)
        diff_ols = abs(w_ols - 2.0)
        diff_tls = abs(w_tls - 2.0)

        # Compare which difference is smaller
        if diff_ols < diff_tls:
            is_tls_winner.append(0)

        elif diff_tls < diff_ols:
            is_tls_winner.append(1)
        else:
            print("Both w_ols and w_tls are equally close to 2.0")

        plt.scatter(X,y)
        plt.title(f"xerr:{x_error_rate:.3f} with yerr:{y_error_rate:.3f}\nOLS:{mse(y,yhat_ols):.3f} and TLS:{mse(y, yhat_tls):.3f}")
        plt.show()
print(f"Total Least Squares leads to better results on average {sum(is_tls_winner)/len(is_tls_winner)}")
