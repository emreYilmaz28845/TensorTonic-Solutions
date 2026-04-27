import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization over last dimension.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)

    x_norm = (x - mean) / np.sqrt(var + eps)

    return gamma * x_norm + beta


def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    Q, K, V shape: (batch_size, seq_len, d_model)
    """
    batch_size, seq_len, d_model = Q.shape

    assert d_model % num_heads == 0
    d_k = d_model // num_heads

    # Linear projections
    Q_proj = Q @ W_q
    K_proj = K @ W_k
    V_proj = V @ W_v

    # Split into heads
    Q_heads = Q_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    K_heads = K_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)
    V_heads = V_proj.reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    # Attention scores
    scores = Q_heads @ K_heads.transpose(0, 1, 3, 2)
    scores = scores / np.sqrt(d_k)

    # Attention weights
    attn_weights = softmax(scores, axis=-1)

    # Weighted sum of values
    context = attn_weights @ V_heads

    # Concatenate heads
    context = context.transpose(0, 2, 1, 3)
    context = context.reshape(batch_size, seq_len, d_model)

    # Output projection
    output = context @ W_o

    return output


def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    Usually: Linear -> ReLU -> Linear
    """
    hidden = x @ W1 + b1
    hidden = np.maximum(0, hidden)   # ReLU
    output = hidden @ W2 + b2

    return output


def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """

    # 1. Self-attention
    attn_output = multi_head_attention(
        x, x, x,
        W_q, W_k, W_v, W_o,
        num_heads
    )

    # 2. Add & Norm
    x = layer_norm(x + attn_output, gamma1, beta1)

    # 3. Feed-forward
    ff_output = feed_forward(x, W1, b1, W2, b2)

    # 4. Add & Norm
    x = layer_norm(x + ff_output, gamma2, beta2)

    return x