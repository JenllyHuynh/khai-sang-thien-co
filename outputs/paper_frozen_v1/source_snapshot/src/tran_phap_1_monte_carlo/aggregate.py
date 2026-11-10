import numpy as np
import pandas as pd

# Đọc toàn bộ các dòng kết quả (mỗi dòng 1 lần chạy) từ CSV
def load_runs(csv_path) -> pd.DataFrame:
    return pd.read_csv(csv_path)

def aggregate_stats(df: pd.DataFrame) -> dict:
    n_runs = len(df)
    total_players_per_group = int(df["n_players_per_group"].sum())

    def _weighted_avg(col: str) -> float:
        return float(np.average(df[col], weights=df["n_players_per_group"]))

    result = {"n_runs": n_runs, "total_players_simulated_per_group": total_players_per_group}

    for phe in ["vo_vi", "tinh_cam"]:
        pct_col = f"{phe}_pct_sharing"
        result[phe] = {
            "pct_sharing_weighted_mean": _weighted_avg(pct_col),
            "pct_sharing_std_across_runs": float(df[pct_col].std()),
            "pct_sharing_min": float(df[pct_col].min()),
            "pct_sharing_max": float(df[pct_col].max()),
            "unique_ratio_weighted_mean": _weighted_avg(f"{phe}_unique_ratio"),
            "max_cluster_size_overall": int(df[f"{phe}_max_cluster"].max()),
        }

    result["ty_le_chenh_lech_trung_binh"] = (
        result["tinh_cam"]["pct_sharing_weighted_mean"] / result["vo_vi"]["pct_sharing_weighted_mean"]
        if result["vo_vi"]["pct_sharing_weighted_mean"] > 0 else float("inf")
    )
    return result
