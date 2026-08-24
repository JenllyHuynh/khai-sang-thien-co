import numpy as np

from .contingency import analyze_independence, build_contingency_table
from .population_model import simulate_population

def repeated_trials_null_distribution(
    n_people: int, p_win: float, p_hoa_baseline: float, n_trials: int = 200, seed_start: int = 0,
) -> np.ndarray:
    diffs = np.zeros(n_trials)
    for i in range(n_trials):
        pop = simulate_population(
            n_people, p_win=p_win, p_hoa_baseline=p_hoa_baseline, true_effect=0.0, seed=seed_start + i
        )
        table = build_contingency_table(pop["a"], pop["b"])
        result = analyze_independence(table)
        diffs[i] = result["chenh_lech_diem_phan_tram"]
    return diffs
