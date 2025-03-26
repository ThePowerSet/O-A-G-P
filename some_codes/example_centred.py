import numpy as np
import tensorflow as tf
import gpflow

# 1) Generate synthetic data
np.random.seed(17)
n = 100
x1 = np.random.uniform(-3, 3, n).reshape(-1, 1)
x2 = np.random.uniform(-3, 3, n).reshape(-1, 1)
X = np.hstack([x1, x2])  # shape [n, 2]

f1 = np.sin(x1).flatten()
f2 = np.cos(x2).flatten()
noise = 0.1 * np.random.randn(n)
Y = (f1 + f2 + noise).reshape(-1, 1)  # shape [n, 1]

# 2) Define a centered kernel
class CenteredKernel(gpflow.kernels.Kernel):
    """
    Wrapper that removes the empirical mean of a base kernel
    computed on the training dataset (X_train) for the active dimensions.
    """
    def __init__(self, base_kernel: gpflow.kernels.Kernel, X_train: np.ndarray):
        # Call the superclass constructor, passing the same active_dims
        super().__init__(active_dims=base_kernel.active_dims)
        
        self.base_kernel = base_kernel
        
        # Convert X_train to tf.float64
        self.X_train = tf.convert_to_tensor(X_train, dtype=tf.float64)
        
        # Compute K_xx on X_train (only on active columns)
        # (If 'active_dims' = [0], only the 0th column will be used, etc.)
        K_xx = self.base_kernel(self.X_train, self.X_train)
        
        # Compute row/column means and the total mean
        self.K_mean_col = tf.reduce_mean(K_xx, axis=0, keepdims=True)  # shape [1, n]
        self.K_mean_row = tf.reduce_mean(K_xx, axis=1, keepdims=True)  # shape [n, 1]
        self.K_mean_total = tf.reduce_mean(K_xx)                       # scalar

    def K(self, X, X2=None):
        """
        Centered kernel = K_raw - (mean_X + mean_X2^T - mean_total)
        """
        if X2 is None:
            X2 = X
        
        # Extract active columns from X and X2
        X_  = X[...,  self.active_dims]
        X2_ = X2[..., self.active_dims]
        
        # Raw kernel
        K_raw = self.base_kernel(X_, X2_)

        # Compute row/column means relative to the training set
        # (still using the same active dimensions)
        mean_X  = tf.reduce_mean(self.base_kernel(X_,  self.X_train[..., self.active_dims]), axis=1, keepdims=True)
        mean_X2 = tf.reduce_mean(self.base_kernel(X2_, self.X_train[..., self.active_dims]), axis=1, keepdims=True)

        # Build the correction matrix
        correction = mean_X + tf.transpose(mean_X2) - self.K_mean_total
        
        return K_raw - correction

    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))

# 3) Define base kernels on x1 and x2 dimensions
base_k1 = gpflow.kernels.SquaredExponential(active_dims=[0])
base_k2 = gpflow.kernels.SquaredExponential(active_dims=[1])

# 4) Create centered versions of these kernels
k1 = CenteredKernel(base_k1, X)  # centered on column 0
k2 = CenteredKernel(base_k2, X)  # centered on column 1
k_total = k1 + k2

# 5) Define the GPR model
model = gpflow.models.GPR(data=(X, Y), kernel=k_total, mean_function=None)

# 6) Optimize using SciPy
opt = gpflow.optimizers.Scipy()
result = opt.minimize(
    model.training_loss, 
    model.trainable_variables, 
    options={"maxiter": 100}
)

print("Optimization result:", result)
print("\nModel parameters after optimization:")
gpflow.utilities.print_summary(model)

