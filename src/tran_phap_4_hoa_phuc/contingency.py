import numpy as np
from scipy import stats as scipy_stats

def build_contingency_table(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    n_a1_b1 = int(np.sum(a & b))
    n_a1_b0 = int(np.sum(a & ~b))
    n_a0_b1 = int(np.sum(~a & b))
    n_a0_b0 = int(np.sum(~a & ~b))
    return np.array([[n_a1_b1, n_a1_b0], [n_a0_b1, n_a0_b0]])

def analyze_independence(table: np.ndarray) -> dict:
    chi2, p_value, dof, _expected = scipy_stats.chi2_contingency(table, correction=True)

    n_a1_b1, n_a1_b0 = table[0]
    n_a0_b1, n_a0_b0 = table[1]
    n_a1 = int(n_a1_b1 + n_a1_b0)
    n_a0 = int(n_a0_b1 + n_a0_b0)

    p_b_given_a1 = n_a1_b1 / n_a1 if n_a1 > 0 else float("nan")
    p_b_given_a0 = n_a0_b1 / n_a0 if n_a0 > 0 else float("nan")

    # Hiệu chỉnh liên tục (+0.5) để tránh chia cho 0 khi 1 ô có count = 0
    odds_ratio = ((n_a1_b1 + 0.5) * (n_a0_b0 + 0.5)) / ((n_a1_b0 + 0.5) * (n_a0_b1 + 0.5))
    relative_risk = p_b_given_a1 / p_b_given_a0 if p_b_given_a0 > 0 else float("nan")

    return {
        "chi2": float(chi2),
        "p_value": float(p_value),
        "dof": int(dof),
        "n_a1": n_a1,
        "n_a0": n_a0,
        "p_hoa_given_trung": float(p_b_given_a1),
        "p_hoa_given_khong_trung": float(p_b_given_a0),
        "chenh_lech_diem_phan_tram": float((p_b_given_a1 - p_b_given_a0) * 100),
        "odds_ratio": float(odds_ratio),
        "relative_risk": float(relative_risk),
        "co_y_nghia_thong_ke": bool(p_value < 0.05),
    }
