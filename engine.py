import numpy as np

class DeltaOptimizer:
    def __init__(self, params, lr=0.005):
        """
        params: List of tuples (param_array, grad_array)
        Eliminates dict lookups.
        """
        self.params = params
        self.lr = np.float32(lr)
        self.beta1 = np.float32(0.9)
        self.beta2 = np.float32(0.999)
        self.eps = np.float32(1e-8)
        self.wd = np.float32(1e-4)
        
        # Pre-allocate moment buffers
        self.m = [np.zeros_like(p) for p, g in params]
        self.v = [np.zeros_like(p) for p, g in params]
        self.t = 0

    def step(self, current_lr=None):
        self.t += 1
        lr = np.float32(current_lr) if current_lr is not None else self.lr
        t_idx = self.t
        
        # Pre-calculate bias corrections
        # Using float32 casting to prevent upcasting
        bias_m = np.float32(1.0) / (np.float32(1.0) - self.beta1 ** t_idx)
        bias_v = np.float32(1.0) / (np.float32(1.0) - self.beta2 ** t_idx)

        # The Hot Loop: Iterate over list (No dicts, no strings)
        for i, (param, grad) in enumerate(self.params):
            # Weight Decay
            g = grad + self.wd * param
            
            # Gradient Clipping
            g = np.clip(g, -1.0, 1.0)

            # Update Moments
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g**2)

            # Apply Update (In-Place)
            m_hat = self.m[i] * bias_m
            v_hat = self.v[i] * bias_v
            
            param -= lr * m_hat / (np.sqrt(v_hat) + self.eps)
            
            # 5. Zero out gradient for next step
            grad.fill(0)
