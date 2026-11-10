import numpy as np

from ..tran_phap_1_monte_carlo.lottery_config import N_NUMBERS, PICK
from .data_generator import generate_draw_history
from .features import build_feature_table
from .overfit_model import predict_top6_per_draw, train_dao_si_khiem_ton, train_lao_tac_ai

#  n_draws mặc định 1560 ~ 10 năm quay số kiểu 3 lần/tuần (52 tuần x 10 năm x 3)
def run_simulation(n_draws: int = 1560, window: int = 52, test_frac: float = 0.1, seed: int = 42) -> dict:
    draws = generate_draw_history(n_draws, seed=seed)
    df = build_feature_table(draws, window=window)

    unique_draw_idx = np.sort(df["draw_index"].unique())
    n_usable = len(unique_draw_idx)
    split_at = unique_draw_idx[int(n_usable * (1 - test_frac))]

    df_train = df[df["draw_index"] < split_at]
    df_test = df[df["draw_index"] >= split_at]

    lao_tac = train_lao_tac_ai(df_train)
    dao_si = train_dao_si_khiem_ton(df_train)

    results = {}
    for trained in (lao_tac, dao_si):
        results[trained.name] = {
            "train": predict_top6_per_draw(trained, df_train),
            "test": predict_top6_per_draw(trained, df_test),
        }

    theoretical_random_matches = PICK * PICK / N_NUMBERS  # E[số trúng] nếu đoán mò ngẫu nhiên

    return {
        "results": results,
        "n_draws": n_draws,
        "window": window,
        "n_train_draws": int(df_train["draw_index"].nunique()),
        "n_test_draws": int(df_test["draw_index"].nunique()),
        "theoretical_random_matches": theoretical_random_matches,
    }
