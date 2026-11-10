import numpy as np
import pandas as pd

from ..tran_phap_1_monte_carlo.lottery_config import N_NUMBERS

def build_feature_table(draws: np.ndarray, window: int = 52) -> pd.DataFrame:
    n_draws = draws.shape[0]
    numbers = np.arange(1, N_NUMBERS + 1)

    # presence[t, n-1] = 1 nếu số n xuất hiện ở lần quay t
    presence = np.zeros((n_draws, N_NUMBERS), dtype=np.int64)
    row_idx = np.repeat(np.arange(n_draws), draws.shape[1])
    col_idx = (draws.astype(np.int64) - 1).reshape(-1)
    presence[row_idx, col_idx] = 1

    # Tần suất trong cửa sổ gần đây & tần suất lũy kế (vector hóa bằng prefix-sum)
    # cumsum[t] = tổng presence từ lần quay 0 đến t-1 (KHÔNG bao gồm lần quay t -> tránh rò rỉ tương lai)
    cumsum = np.vstack([np.zeros((1, N_NUMBERS), dtype=np.int64), presence.cumsum(axis=0)])

    # Khoảng cách tới lần xuất hiện gần nhất TRƯỚC t (vector hóa bằng cumulative-max)
    time_idx = np.arange(n_draws)
    occurred_idx = np.where(presence == 1, time_idx[:, None], -1)
    cummax = np.maximum.accumulate(occurred_idx, axis=0)  # cummax[t] = lần gần nhất <= t đã xuất hiện
    last_seen_before = np.vstack([np.full((1, N_NUMBERS), -1, dtype=np.int64), cummax[:-1]])

    # Chỉ giữ các lần quay có đủ `window` lịch sử phía trước để tính đặc trưng
    valid_t = np.arange(window, n_draws)
    if valid_t.size == 0:
        raise ValueError(
            f"n_draws ({n_draws}) phải lớn hơn window ({window}) để có đủ lịch sử tính đặc trưng."
        )

    freq_last_window = (cumsum[valid_t] - cumsum[valid_t - window]) / window
    overall_freq = cumsum[valid_t] / valid_t[:, None]
    gap = last_seen_before[valid_t]
    gap = np.where(gap < 0, window, np.minimum(valid_t[:, None] - gap, window)) / window

    n_valid = valid_t.shape[0]
    df = pd.DataFrame({
        "draw_index": np.repeat(valid_t, N_NUMBERS),
        "number": np.tile(numbers, n_valid),
        "freq_last_window": freq_last_window.reshape(-1),
        "overall_freq": overall_freq.reshape(-1),
        "gap_since_last": gap.reshape(-1),
        "label": presence[valid_t].reshape(-1),
    })
    return df
