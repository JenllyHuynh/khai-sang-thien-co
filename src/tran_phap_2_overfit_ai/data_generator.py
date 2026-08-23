import numpy as np

from ..tran_phap_1_monte_carlo.lottery_config import PICK  # noqa: F401 (re-export tiện dùng)
from ..tran_phap_1_monte_carlo.ticket_generator import generate_uniform

def generate_draw_history(n_draws: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draws = generate_uniform(n_draws, rng)
    return np.sort(draws, axis=1)
