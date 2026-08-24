import numpy as np

# Linear Congruential Generator kiểu RANDU (IBM, 1968)
class WeakLCG:

    MODULUS = 2 ** 31
    MULTIPLIER = 65539
    INCREMENT = 0

    def __init__(self, seed: int):
        # RANDU yêu cầu seed lẻ để tránh suy biến về 0
        self.state = (seed % self.MODULUS) | 1

    def next(self) -> int:
        self.state = (self.MULTIPLIER * self.state + self.INCREMENT) % self.MODULUS
        return self.state

    def next_float(self) -> float:
        return self.next() / self.MODULUS

    def generate(self, n: int) -> np.ndarray:
        return np.array([self.next_float() for _ in range(n)])

# Sinh n số từ LCG yếu (RANDU) - 'yêu quái tầm thường'
def generate_weak_lcg(n: int, seed: int = 1) -> np.ndarray:
    return WeakLCG(seed).generate(n)

# Sinh n số từ CSPRNG mạnh - numpy mặc định dùng PCG64, vượt qua hầu hết các bộ kiểm định thống kê (NIST SP 800-22, Dieharder...)
def generate_strong_csprng(n: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random(n)

def attempt_state_reconstruction_crack(observed: np.ndarray) -> dict:
    observed_states = np.round(observed * WeakLCG.MODULUS).astype(np.int64) % WeakLCG.MODULUS
    reconstructed = WeakLCG(seed=1)
    reconstructed.state = int(observed_states[0]) or 1  # tránh state=0 (suy biến)

    n_predict = len(observed) - 1
    predicted = np.array([reconstructed.next_float() for _ in range(n_predict)])
    actual = observed[1:]

    abs_errors = np.abs(predicted - actual)
    max_abs_error = float(abs_errors.max()) if n_predict else 0.0
    mean_abs_error = float(abs_errors.mean()) if n_predict else 0.0

    return {
        "n_predicted": n_predict,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "cracked": max_abs_error < 1e-9,
    }
