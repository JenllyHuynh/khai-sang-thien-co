from .contingency import analyze_independence, build_contingency_table
from .narrative_bias import repeated_trials_null_distribution
from .population_model import simulate_population

def run_simulation(
    n_people: int = 2_000_000,
    p_win: float = 0.001,
    p_hoa_baseline: float = 0.05,
    n_trials_null: int = 200,
    seed: int = 42,
) -> dict:
    pop = simulate_population(n_people, p_win=p_win, p_hoa_baseline=p_hoa_baseline, true_effect=0.0, seed=seed)
    table = build_contingency_table(pop["a"], pop["b"])
    independence_result = analyze_independence(table)

    # Quần thể nhỏ hơn cho vòng lặp nhiều lần (n_trials_null lần) - vẫn đủ
    # lớn để thống kê có ý nghĩa, nhưng không lặp lại việc mô phỏng hàng
    # triệu người hàng trăm lần một cách lãng phí.
    n_people_per_trial = max(n_people // 10, 50_000)
    null_diffs = repeated_trials_null_distribution(
        n_people=n_people_per_trial,
        p_win=p_win,
        p_hoa_baseline=p_hoa_baseline,
        n_trials=n_trials_null,
        seed_start=seed,
    )

    return {
        "table": table,
        "independence_result": independence_result,
        "null_diffs": null_diffs,
        "params": {
            "n_people": n_people,
            "p_win": p_win,
            "p_hoa_baseline": p_hoa_baseline,
            "n_trials_null": n_trials_null,
            "n_people_per_trial": n_people_per_trial,
        },
    }
