import numpy as np

from src.tran_phap_4_hoa_phuc.contingency import analyze_independence, build_contingency_table
from src.tran_phap_4_hoa_phuc.narrative_bias import repeated_trials_null_distribution
from src.tran_phap_4_hoa_phuc.population_model import simulate_population
from src.tran_phap_4_hoa_phuc.simulate import run_simulation
from src.tran_phap_4_hoa_phuc.stats import summarize

def test_simulate_population_shapes_and_rates():
    pop = simulate_population(200_000, p_win=0.01, p_hoa_baseline=0.05, seed=1)
    assert pop["a"].shape == (200_000,)
    assert pop["b"].shape == (200_000,)

    # Tỷ lệ trúng số thực tế phải gần với p_win đã đặt (dung sai thống kê)
    assert abs(pop["a"].mean() - 0.01) < 0.005
    assert abs(pop["b"].mean() - 0.05) < 0.01

def test_build_contingency_table_sums_to_n():
    pop = simulate_population(50_000, p_win=0.01, p_hoa_baseline=0.05, seed=2)
    table = build_contingency_table(pop["a"], pop["b"])

    assert table.sum() == 50_000
    assert table.shape == (2, 2)

def test_independence_holds_when_true_effect_zero():
    pop = simulate_population(1_000_000, p_win=0.005, p_hoa_baseline=0.05, true_effect=0.0, seed=3)
    table = build_contingency_table(pop["a"], pop["b"])
    result = analyze_independence(table)

    diff = abs(result["p_hoa_given_trung"] - result["p_hoa_given_khong_trung"])
    assert diff < 0.02  # chênh lệch nhỏ, trong biên độ nhiễu thống kê hợp lý
    assert 0.5 < result["odds_ratio"] < 2.0  # gần 1.0


def test_true_effect_creates_detectable_difference():
    pop = simulate_population(500_000, p_win=0.01, p_hoa_baseline=0.05, true_effect=0.3, seed=4)
    table = build_contingency_table(pop["a"], pop["b"])
    result = analyze_independence(table)

    assert result["p_hoa_given_trung"] > result["p_hoa_given_khong_trung"] + 0.1
    assert result["p_value"] < 0.01
    assert result["co_y_nghia_thong_ke"] is True


def test_repeated_trials_null_distribution_centered_near_zero():
    diffs = repeated_trials_null_distribution(
        n_people=100_000, p_win=0.01, p_hoa_baseline=0.05, n_trials=60, seed_start=0
    )
    assert diffs.shape == (60,)
    # Trung bình của phân phối "nhiễu do may rủi" phải gần 0
    assert abs(diffs.mean()) < 1.0

def test_run_simulation_and_summarize_end_to_end():
    result = run_simulation(n_people=300_000, p_win=0.01, p_hoa_baseline=0.05, n_trials_null=30, seed=5)
    summary = summarize(result)

    assert "chi2" in summary and "p_value" in summary
    assert summary["table"][0][0] + summary["table"][0][1] == summary["n_a1"]
    assert summary["table"][1][0] + summary["table"][1][1] == summary["n_a0"]
    assert isinstance(summary["co_y_nghia_thong_ke"], bool)
