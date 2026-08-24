import numpy as np

def summarize(sim_result: dict) -> dict:
    ir = sim_result["independence_result"]
    null_diffs = sim_result["null_diffs"]

    return {
        "params": sim_result["params"],
        "table": sim_result["table"].tolist(),
        "chi2": ir["chi2"],
        "p_value": ir["p_value"],
        "co_y_nghia_thong_ke": ir["co_y_nghia_thong_ke"],
        "n_a1": ir["n_a1"],
        "n_a0": ir["n_a0"],
        "p_hoa_given_trung": ir["p_hoa_given_trung"],
        "p_hoa_given_khong_trung": ir["p_hoa_given_khong_trung"],
        "chenh_lech_diem_phan_tram": ir["chenh_lech_diem_phan_tram"],
        "odds_ratio": ir["odds_ratio"],
        "relative_risk": ir["relative_risk"],
        "null_distribution_mean": float(np.mean(null_diffs)),
        "null_distribution_std": float(np.std(null_diffs)),
        "null_distribution_min": float(np.min(null_diffs)),
        "null_distribution_max": float(np.max(null_diffs)),
    }
