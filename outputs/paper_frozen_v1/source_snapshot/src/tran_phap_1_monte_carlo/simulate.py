from math import comb

import numpy as np

from .encode import encode_tickets
from .lottery_config import N_NUMBERS, PICK
from .ticket_generator import generate_bias, generate_uniform

try:
    from tqdm import tqdm
except ImportError:  # tqdm là tiện ích, không bắt buộc phải có
    def tqdm(iterable=None, total=None, desc=None, unit=None):
        return iterable if iterable is not None else range(0)

def generate_ids_in_chunks(n_total: int, chunk_size: int, generator_fn, rng,
                            desc: str = "", verbose: bool = True) -> np.ndarray:
    parts = []
    remaining = n_total
    done = 0
    while remaining > 0:
        batch = min(chunk_size, remaining)
        tickets = generator_fn(batch, rng)
        parts.append(encode_tickets(tickets))
        remaining -= batch
        done += batch
        if verbose:
            print(f"  [{desc}] {done:,}/{n_total:,} vé", end="\r")
    if verbose:
        print()
    return np.concatenate(parts)

# Giữ tên cũ để tương thích ngược, vì còn import _generate_ids_in_chunks
_generate_ids_in_chunks = generate_ids_in_chunks

def run_simulation(n_players: int = 2_000_000, chunk_size: int = 200_000, seed: int = 42) -> dict:

    rng = np.random.default_rng(seed)

    ids_vo_vi = generate_ids_in_chunks(n_players, chunk_size, generate_uniform, rng, "Vô Vi")
    ids_tinh_cam = generate_ids_in_chunks(n_players, chunk_size, generate_bias, rng, "Tình Cảm")

    total_combos = comb(N_NUMBERS, PICK)
    theoretical_p = 1 / total_combos

    return {
        "ids_vo_vi": ids_vo_vi,
        "ids_tinh_cam": ids_tinh_cam,
        "n_players": n_players,
        "total_combos": total_combos,
        "theoretical_win_probability": theoretical_p,
    }
