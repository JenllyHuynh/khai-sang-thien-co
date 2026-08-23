"""
Xây đặc trưng (feature) từ lịch sử quay số để "nhồi" cho Lão Tặc AI học.

Toàn bộ tính toán được VECTOR HÓA bằng numpy (prefix-sum cho tần suất,
cumulative-max cho khoảng cách xuất hiện gần nhất) — không có Python
for-loop nào chạy qua từng cặp (lần quay, số), dù dữ liệu là hàng chục
nghìn dòng.
"""
import numpy as np
import pandas as pd

from ..tran_phap_1_monte_carlo.lottery_config import N_NUMBERS


def build_feature_table(draws: np.ndarray, window: int = 52) -> pd.DataFrame:
    """draws: (n_draws, PICK) đã sort mỗi hàng, thứ tự thời gian tăng dần.

    Trả về DataFrame dạng "long" — mỗi dòng là 1 cặp (lần quay t, số n),
    kèm các cột đặc trưng + nhãn (label = 1 nếu số n xuất hiện ở lần quay t).

    Cột `draw_index` và `number` CỐ TÌNH được đưa vào làm đặc trưng cho
    "Lão Tặc AI" — đây chính là "con dao 2 lưỡi" khiến nó overfit: cặp
    (draw_index, number) là ĐỊNH DANH DUY NHẤT của mỗi dòng dữ liệu, không
    phải tín hiệu dự đoán thật. Một cây quyết định không giới hạn độ sâu
    có thể lợi dụng cặp định danh này để "học thuộc lòng" đáp án của toàn
    bộ lịch sử, thay vì học quy luật thật (vì làm gì có quy luật thật
    trong random thuần túy).
    """
    n_draws = draws.shape[0]
    numbers = np.arange(1, N_NUMBERS + 1)

    # presence[t, n-1] = 1 nếu số n xuất hiện ở lần quay t
    presence = np.zeros((n_draws, N_NUMBERS), dtype=np.int64)
    row_idx = np.repeat(np.arange(n_draws), draws.shape[1])
    col_idx = (draws.astype(np.int64) - 1).reshape(-1)
    presence[row_idx, col_idx] = 1

    # --- Tần suất trong cửa sổ gần đây & tần suất lũy kế (vector hóa bằng prefix-sum) ---
    # cumsum[t] = tổng presence từ lần quay 0 đến t-1 (KHÔNG bao gồm lần quay t -> tránh rò rỉ tương lai)
    cumsum = np.vstack([np.zeros((1, N_NUMBERS), dtype=np.int64), presence.cumsum(axis=0)])

    # --- Khoảng cách tới lần xuất hiện gần nhất TRƯỚC t (vector hóa bằng cumulative-max) ---
    time_idx = np.arange(n_draws)
    occurred_idx = np.where(presence == 1, time_idx[:, None], -1)
    cummax = np.maximum.accumulate(occurred_idx, axis=0)  # cummax[t] = lần gần nhất <= t đã xuất hiện
    last_seen_before = np.vstack([np.full((1, N_NUMBERS), -1, dtype=np.int64), cummax[:-1]])

    # --- Chỉ giữ các lần quay có đủ `window` lịch sử phía trước để tính đặc trưng ---
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
