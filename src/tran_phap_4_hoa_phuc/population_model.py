import numpy as np

# Sinh quần thể n_people người, mỗi người có 2 biến cố nhị phân A (trúng số) và B (gặp họa)
def simulate_population(
    n_people: int,
    p_win: float = 0.001,
    p_hoa_baseline: float = 0.05,
    true_effect: float = 0.0, # cộng thêm vào P(B|A) so với mức nền P(B)
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)

    a = rng.random(n_people) < p_win

    p_hoa_given_a = float(np.clip(p_hoa_baseline + true_effect, 0.0, 1.0))
    p_hoa_per_person = np.where(a, p_hoa_given_a, p_hoa_baseline)
    b = rng.random(n_people) < p_hoa_per_person

    return {
        "a": a,
        "b": b,
        "n_people": n_people,
        "p_win": p_win,
        "p_hoa_baseline": p_hoa_baseline,
        "true_effect": true_effect,
    }
