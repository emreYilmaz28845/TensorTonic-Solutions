import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    pe = np.zeros((seq_length, d_model))
    # Your code here
    for i in range(seq_length):
        for j in range(d_model):
            if j % 2 == 0:
                pe[i, j] = np.sin(i / (10000 ** (j / d_model)))
            else:
                pe[i, j] = np.cos(i / (10000 ** ((j - 1) / d_model)))
    return pe