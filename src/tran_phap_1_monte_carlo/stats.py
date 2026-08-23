import numpy as np
import pandas as pd

def cluster_stats(ids: np.ndarray) -> dict:
    unique, counts = np.unique(ids, return_counts=True)
    n_players = int(ids.shape[0])
    n_unique = int(unique.shape[0])
    duplicated_mask = counts > 1
    n_players_in_dup_cluster = int(counts[duplicated_mask].sum())

    return {
        "n_players": n_players,
        "n_unique_combos": n_unique,
        "unique_ratio": n_unique / n_players,
        "pct_players_sharing_ticket": 100 * n_players_in_dup_cluster / n_players,
        "max_cluster_size": int(counts.max()),
        "mean_cluster_size_if_shared": float(counts[duplicated_mask].mean()) if duplicated_mask.any() else 0.0,
        "_unique": unique,
        "_counts": counts,
    }

# Trả về DataFrame top N tổ hợp được nhiều người chọn nhất
def top_popular_combos(stats: dict, decode_fn, top_n: int = 15) -> pd.DataFrame:
    unique = stats["_unique"]
    counts = stats["_counts"]
    order = np.argsort(-counts)[:top_n]
    rows = [
        {"to_hop": decode_fn(unique[idx]), "so_nguoi_chon": int(counts[idx])}
        for idx in order
    ]
    return pd.DataFrame(rows)
