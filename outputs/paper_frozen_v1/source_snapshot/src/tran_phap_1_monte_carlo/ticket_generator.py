import numpy as np

from .lottery_config import N_NUMBERS, PICK, BIRTHDAY_MAX, LUCKY_NUMBERS, BiasConfig

# Phe Vô Vi: mọi số có trọng số như nhau -> random thuần túy
def _weights_vo_vi() -> np.ndarray:
    return np.ones(N_NUMBERS, dtype=np.float32)

# Phe Tình Cảm: ưu tiên số <=31 (ngày sinh) và các số hên.
def _weights_tinh_cam(cfg: BiasConfig) -> np.ndarray:
    numbers = np.arange(1, N_NUMBERS + 1)
    w = np.full(N_NUMBERS, cfg.base_weight, dtype=np.float32)
    w[numbers <= BIRTHDAY_MAX] = cfg.birthday_weight
    for lucky in LUCKY_NUMBERS:
        w[lucky - 1] = cfg.lucky_weight
    return w

def generate_tickets(n: int, weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    u = rng.random((n, N_NUMBERS), dtype=np.float32)
    u = np.clip(u, 1e-12, 1.0)  # tránh log(0)/chia 0
    keys = u ** (1.0 / weights)  # broadcast (n,55) ** (55,)

    # Lấy PICK cột có key lớn nhất mỗi hàng - không cần sort toàn phần,
    # argpartition nhanh hơn argsort đáng kể khi PICK << N_NUMBERS.
    top_idx = np.argpartition(-keys, PICK - 1, axis=1)[:, :PICK]
    numbers = (top_idx + 1).astype(np.uint8)  # số 1..55
    return numbers

# Sinh vé cho Phe Vô Vi (random thuần túy)
def generate_uniform(n: int, rng: np.random.Generator) -> np.ndarray:
    return generate_tickets(n, _weights_vo_vi(), rng)

# Sinh vé cho Phe Tình Cảm (thiên vị số đẹp/ngày sinh)
def generate_bias(n: int, rng: np.random.Generator, cfg: BiasConfig = BiasConfig()) -> np.ndarray:
    return generate_tickets(n, _weights_tinh_cam(cfg), rng)
