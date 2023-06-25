import torch
import numpy as np


def init_mask(Q, K, pad_idx=1, method="padding"):
    """
    Q : (num_batch, seq_len)
    K : (num_batch, seq_len)
    """
    Q_seq_len, K_seq_len = Q.shape[1], K.shape[1]
    
    assert method in ["padding", "tril"], "Not supported method."
    
    if method == "padding":
        key_mask = K.ne(pad_idx).unsqueeze(1).unsqueeze(2)  # (num_batch, 1, 1, K_seq_len)
        key_mask = key_mask.repeat(1, 1, Q_seq_len, 1)  # (num_batch, 1, Q_seq_len, K_seq_len)
        
        query_mask = Q.ne(pad_idx).unsqueeze(1).unsqueeze(3)  # (num_batch, 1, 1, Q_seq_len)
        query_mask = query_mask.repeat(1, 1, 1, K_seq_len)  # (num_batch, 1, Q_seq_len, K_seq_len)
        
        return key_mask & query_mask
    elif method == "tril":
        tril = np.tril(
            np.ones(
                shape=(
                    Q_seq_len, K_seq_len
                )
            ),
            k=0
        ).astype("uint8")
        return torch.tensor(
            tril,
            dtype=torch.bool, 
            requires_grad=False
        )