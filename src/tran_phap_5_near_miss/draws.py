import numpy as np
from math import comb

from .config import N_NUMBERS, PICK


def pmf_so_trung() -> np.ndarray:
    # P(trùng k số), k=0..6, phân phối siêu bội cho một vé cố định.
    total = comb(N_NUMBERS, PICK)
    return np.array([
        comb(PICK, k) * comb(N_NUMBERS - PICK, PICK - k) / total for k in range(PICK + 1)
    ])


def sample_so_trung(rng: np.random.Generator, shape) -> np.ndarray:
    return rng.hypergeometric(PICK, N_NUMBERS - PICK, PICK, size=shape).astype(np.int8)


def quay_so_that(rng: np.random.Generator) -> np.ndarray:
    return np.sort(rng.choice(np.arange(1, N_NUMBERS + 1), size=PICK, replace=False))


def so_trung(ve: np.ndarray, ket_qua: np.ndarray) -> int:
    return int(np.intersect1d(ve, ket_qua).size)


def dung_ky_quay_co_k_trung(rng: np.random.Generator, ve: np.ndarray, k: int) -> np.ndarray:
    ve = np.asarray(ve)
    ngoai = np.setdiff1d(np.arange(1, N_NUMBERS + 1), ve)
    trong = rng.choice(ve, size=k, replace=False) if k > 0 else np.array([], dtype=int)
    khac = rng.choice(ngoai, size=PICK - k, replace=False) if k < PICK else np.array([], dtype=int)
    return np.sort(np.concatenate([trong, khac]))
