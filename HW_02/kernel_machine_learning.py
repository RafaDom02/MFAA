from typing import Callable, Tuple

import numpy as np
from scipy.spatial import distance


def linear_kernel(
    X: np.ndarray,
    X_prime: np.ndarray,
) -> np.ndarray:
    return X @ X_prime.T


def exponential_kernel(
    X: np.ndarray,
    X_prime: np.ndarray,
    A: float,
    l: float
) -> np.ndarray:
    d = distance.cdist(X, X_prime, metric='minkovski', p=1.0)
    return A * np.exp(- d / l)


def rbf_kernel(
    X: np.ndarray,
    X_prime: np.ndarray,
    A: float,
    ls: float,
) -> np.ndarray:
    """
    Parameters
    ----------
    X:
        Data matrix
    X_prime:
        Data matrix
    A:
        Output variance
    ls:
        Kernel lengthscale

    Returns
    -------
    kernel matrix

    Notes
    -------
    Alternative parametrization (e.g. en sklearn)
    gamma = 0.5 / ls**2

    Example
    -------
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> import gaussian_process_regression as gp
    >>> X = np.array([[1,2], [3, 4], [5,6]])
    >>> X_prime = np.array([[1,2], [3, 4]])
    >>> A, l = 3, 10.0
    >>> kernel_matrix = gp.rbf_kernel(X, X_prime, A, l)
    >>> print(kernel_matrix)
    """
    d = distance.cdist(X, X_prime, metric='euclidean')
    return A * np.exp(-0.5 * (d / ls)**2)


def kernel_pca(
    X: np.ndarray,
    X_test: np.ndarray,
    kernel: Callable,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Parameters
    ----------
    X:
        Data matrix
    X_test:
        data matrix
    A:
        output variance
    ls:
        kernel lengthscale

    Returns
    -------
    X_test_hat:
        Projection of X_test on the principal components
    lambda_eigenvals:
        Eigenvalues of the centered kernel
    alpha_eigenvecs:
        Principal components. These are the eigenvectors
        of the centered kernel with the RKHS normalization

    Notes
    -------
    In the corresponding method of sklearn the eigenvectors
    are normalized in l2.

    """

    # Calculamos la matriz de kernel y sus medias
    K = kernel(X, X)
    K_mean_row = np.mean(K, axis=0, keepdims=True)  # Media por columnas
    K_mean_col = np.mean(K, axis=1, keepdims=True)  # Media por filas
    K_mean = np.mean(K)                             # Media global de la matriz

    # Centramos la matriz de kernel
    K_tilda = K - K_mean_row - K_mean_col + K_mean

    # Hacemos el SVD de la matriz de kernel
    alpha_eigenvecs, lambda_eigenvals, _ = np.linalg.svd(K_tilda, full_matrices=False)

    # Normalizamos los eigenvectores
    lambda_sqrt = np.sqrt(lambda_eigenvals)         # Raíz cuadrada de los valores propios
    alpha_eigenvecs /= lambda_sqrt[np.newaxis, :]   # Normalización por columnas

    # Calculamos la matriz de kernel para los datos de prueba
    K_test = kernel(X_test, X)

    # Centramos la matriz de kernel de prueba
    K_test_mean_row = np.mean(K_test, axis=1, keepdims=True)    # Media por filas en X_test
    K_test_mean_col = np.mean(K, axis=0, keepdims=True)         # Media por columnas en X
    K_tilda_test = K_test - K_test_mean_row - K_test_mean_col + K_mean

    # Proyectamos los datos de prueba en los componentes principales
    X_test_hat = K_tilda_test @ alpha_eigenvecs

    return X_test_hat, lambda_eigenvals, alpha_eigenvecs
