import random
import numpy as np

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Fix every source of randomness we control."""
    random.seed(seed)
    np.random.seed(seed)
