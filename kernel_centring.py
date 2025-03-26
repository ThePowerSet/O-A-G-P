import numpy as np
import tensorflow as tf
import gpflow

def fbm_kernel_numpy(x, y, gamma=0.5, alpha=1.0): # this doesn't lead to a meaningful kernel, dk why
    """
    Computes the fBM kernel matrix between vectors x and y.
    Args:
        x (np.ndarray): shape (n, 1)
        y (np.ndarray): shape (m, 1)
        gamma (float): 0 < gamma < 1
        alpha (float): scale factor
    Returns:
        np.ndarray: Gram matrix (n x m)
    """
    x = np.asarray(x).flatten()[:, None]  # shape (n, 1)
    y = np.asarray(y).flatten()[None, :]  # shape (1, m)

    x2g = np.power(np.abs(x), 2 * gamma)  # shape (n, 1)
    y2g = np.power(np.abs(y), 2 * gamma)  # shape (1, m)
    dist = np.abs(x - y)                  # shape (n, m)
    diff2g = np.power(dist, 2 * gamma)

    return 0.5 * alpha**2 * (x2g + y2g - diff2g)

def centre_gram_matrix(K):
    """
    Empirically center a Gram matrix: K_c = C K C
    Args:
        K (np.ndarray): Gram matrix of shape (n, n)
    Returns:
        Centred Gram matrix K_c
    """
    n = K.shape[0]
    I = np.eye(n)
    one = np.ones((n, 1))
    C = I - (1 / n) * (one @ one.T)
    K_centered = C @ K @ C
    return K_centered

# === EXAMPLE ===
# Input data: weights of 3 individuals
weights = np.array([[1], [2], [3]])

Kbm = fbm_kernel_numpy(weights, weights, gamma=0.5, alpha=1.0)

kernel = gpflow.kernels.SquaredExponential(lengthscales=5.0)
Kse = kernel(weights).numpy()

Kbm_c = centre_gram_matrix(Kbm)
Kse_c = centre_gram_matrix(Kse)







