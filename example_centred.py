import numpy as np
import tensorflow as tf
import gpflow
from gpflow.models import GPR
from gpflow.optimizers import Scipy
from gpflow.utilities import print_summary
import matplotlib.pyplot as plt

# 1. Dati simulati
np.random.seed(42)
n = 100
x1 = np.random.uniform(-3, 3, n).reshape(-1, 1)
x2 = np.random.uniform(-3, 3, n).reshape(-1, 1)
X = np.hstack([x1, x2])
f1 = np.sin(x1).flatten()
f2 = np.cos(x2).flatten()
Y = (f1 + f2 + 0.1 * np.random.randn(n)).reshape(-1, 1)

# 2. Funzione per centrare un Gram matrix
def empirical_centering(X, kernel):
    K = kernel(tf.convert_to_tensor(X, dtype=tf.float64))
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K.numpy() @ H

# 3. Kernel centrato (manuale, fisso)
class CenteredKernel(gpflow.kernels.Kernel):
    def __init__(self, base_kernel, X_full):
        super().__init__()
        self.base_kernel = base_kernel
        self.active_dims = base_kernel.active_dims
        dims = self.active_dims
        if hasattr(dims, "numpy"):
            dims = dims.numpy()
        X_proj = X_full[:, dims]
        self.X_proj = tf.convert_to_tensor(X_proj, dtype=tf.float64)
        self.K_c_np = empirical_centering(self.X_proj, gpflow.kernels.SquaredExponential())
        self.K_c = tf.convert_to_tensor(self.K_c_np, dtype=tf.float64)

    @tf.autograph.experimental.do_not_convert
    def K(self, X, X2=None):
        return self.K_c

    @tf.autograph.experimental.do_not_convert
    def K_diag(self, X):
        return tf.linalg.diag_part(self.K_c)

# 4. Kernel per effetti principali centrati
k1 = gpflow.kernels.SquaredExponential(active_dims=[0])
k2 = gpflow.kernels.SquaredExponential(active_dims=[1])
kc1 = CenteredKernel(k1, X)
kc2 = CenteredKernel(k2, X)

# 5. Modello con somma dei due effetti principali centrati
kernel = gpflow.kernels.Sum([kc1, kc2])
model = GPR(data=(X, Y), kernel=kernel, mean_function=None)

# 6. Stima (attenzione: nessun parametro è ottimizzabile perché i kernel sono fissi)
opt = Scipy()
opt.minimize(model.training_loss, model.trainable_variables)
print_summary(model)

# 7. Predizione e visualizzazione
mean, var = model.predict_f(X)

plt.figure(figsize=(12, 5))
plt.plot(mean, label="Posterior mean")
plt.plot(Y, label="Observed (true + noise)", alpha=0.5)
plt.title("Additive GP – main effects only")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

