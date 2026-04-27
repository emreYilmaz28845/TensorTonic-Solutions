import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Expected shapes:
    Q, K, V: (batch_size, seq_len, d_model)
    W_q, W_k, W_v, W_o: (d_model, d_model)
    
    Output:
    (batch_size, seq_len, d_model)
    """
    
    batch_size, seq_len, d_model = Q.shape
    
    assert d_model % num_heads == 0
    d_k = d_model // num_heads
    
    # 1. Linear projections
    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v
    
    # 2. Split into heads
    # (batch, seq_len, d_model)
    # -> (batch, seq_len, num_heads, d_k)
    # -> (batch, num_heads, seq_len, d_k)
    Q_heads = Q_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K_heads = K_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V_heads = V_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    
    # 3. Scaled dot-product attention
    # scores: (batch, num_heads, seq_len, seq_len)
    scores = Q_heads @ K_heads.transpose(0, 1, 3, 2)
    scores = scores / np.sqrt(d_k)
    
    attention_weights = softmax(scores, axis=-1)
    
    # context: (batch, num_heads, seq_len, d_k)
    context = attention_weights @ V_heads
    
    # 4. Concatenate heads
    # (batch, num_heads, seq_len, d_k)
    # -> (batch, seq_len, num_heads, d_k)
    # -> (batch, seq_len, d_model)
    context = context.transpose(0, 2, 1, 3)
    context = context.reshape(batch_size, seq_len, d_model)
    
    # 5. Final output projection
    output = context @ W_o
    
    return output