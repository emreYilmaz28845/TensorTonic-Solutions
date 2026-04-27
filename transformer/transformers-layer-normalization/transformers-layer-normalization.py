import numpy as np

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    x: (..., d_model)
    gamma: (d_model,)
    beta: (d_model,)
    """
    
    # 1. mean over last dimension
    mean = np.mean(x, axis=-1, keepdims=True)
    
    # 2. variance over last dimension
    var = np.var(x, axis=-1, keepdims=True)
    
    # 3. normalize
    x_norm = (x - mean) / np.sqrt(var + eps)
    
    # 4. scale and shift
    out = gamma * x_norm + beta
    
    return out