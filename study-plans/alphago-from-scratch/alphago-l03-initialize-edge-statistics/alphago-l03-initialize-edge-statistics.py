import numpy as np


def initialize_mcts_edges(
    policy_logits: np.ndarray, legal_mask: np.ndarray
) -> dict:
  """Returns: N as an int64 array; W, Q, and P as float64 arrays in a dictionary."""
  # Ensure inputs are numpy arrays
  logits = np.array(policy_logits, dtype=np.float64)
  mask = np.array(legal_mask, dtype=bool)

  n_actions = len(logits)

  # Initialize N as int64 zeros; W, Q, and P as float64 zeros
  N = np.zeros(n_actions, dtype=np.int64)
  W = np.zeros(n_actions, dtype=np.float64)
  Q = np.zeros(n_actions, dtype=np.float64)
  P = np.zeros(n_actions, dtype=np.float64)

  # Calculate P only if there are legal actions
  if np.any(mask):
    # Ignore illegal actions by setting their logits to -inf when finding max
    masked_logits = np.where(mask, logits, -np.inf)
    m = np.max(masked_logits)

    # Compute exp(l_a - m) for legal actions, 0 for illegal actions
    exp_logits = np.where(mask, np.exp(logits - m), 0.0)

    # Sum of exponentials over all legal actions (normalization denominator)
    sum_exp = np.sum(exp_logits)

    if sum_exp > 0:
      P = exp_logits / sum_exp

  return {"N": N, "W": W, "Q": Q, "P": P}