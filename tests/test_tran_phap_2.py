import numpy as np
import pandas as pd

from src.tran_phap_2_overfit_ai.data_generator import generate_draw_history
from src.tran_phap_2_overfit_ai.features import build_feature_table
from src.tran_phap_2_overfit_ai.overfit_model import (
    predict_top6_per_draw,
    train_dao_si_khiem_ton,
    train_lao_tac_ai,
)
from src.tran_phap_2_overfit_ai.simulate import run_simulation
from src.tran_phap_2_overfit_ai.stats import summarize
from src.tran_phap_1_monte_carlo.lottery_config import N_NUMBERS, PICK

def test_generate_draw_history_shape_and_validity():
    draws = generate_draw_history(200, seed=1)
    assert draws.shape == (200, PICK)
    assert draws.min() >= 1 and draws.max() <= N_NUMBERS
    for row in draws[:50]:
        assert len(set(row.tolist())) == PICK
        assert list(row) == sorted(row)  # đã sort tăng dần

def test_build_feature_table_no_future_leakage_and_shape():
    draws = generate_draw_history(200, seed=1)
    window = 20
    df = build_feature_table(draws, window=window)

    # Chỉ có các draw_index >= window (đủ lịch sử để tính đặc trưng)
    assert df["draw_index"].min() == window
    assert df["draw_index"].max() == 199

    # Mỗi draw_index phải có đúng N_NUMBERS dòng (1 dòng / số)
    counts = df.groupby("draw_index").size()
    assert (counts == N_NUMBERS).all()

    # label phải đúng PICK lượt "1" cho mỗi draw_index (vì mỗi lần quay có đúng PICK số)
    label_sums = df.groupby("draw_index")["label"].sum()
    assert (label_sums == PICK).all()

def test_lao_tac_ai_memorizes_training_perfectly():
    draws = generate_draw_history(300, seed=2)
    df = build_feature_table(draws, window=20)

    lao_tac = train_lao_tac_ai(df)
    result = predict_top6_per_draw(lao_tac, df)

    # Trên chính dữ liệu đã học, trung bình số trúng phải rất gần PICK (6/6)
    assert result["so_khop"].mean() >= PICK - 0.5

def test_dao_si_khiem_ton_does_not_memorize():
    draws = generate_draw_history(300, seed=2)
    df = build_feature_table(draws, window=20)

    dao_si = train_dao_si_khiem_ton(df)
    result = predict_top6_per_draw(dao_si, df)

    theoretical = PICK * PICK / N_NUMBERS
    # Cho phép sai số nhưng không được vượt quá xa mức đoán mò
    assert result["so_khop"].mean() < theoretical + 1.0

def test_run_simulation_end_to_end_and_overfit_gap():
    result = run_simulation(n_draws=300, window=20, test_frac=0.15, seed=3)
    summary = summarize(result)

    lao_tac_name = [n for n in summary["models"] if "Học Vẹt" in n][0]
    dao_si_name = [n for n in summary["models"] if "Baseline" in n][0]

    lao_tac_gap = summary["models"][lao_tac_name]["train"]["avg_matches"] - \
        summary["models"][lao_tac_name]["test"]["avg_matches"]
    dao_si_gap = summary["models"][dao_si_name]["train"]["avg_matches"] - \
        summary["models"][dao_si_name]["test"]["avg_matches"]

    assert lao_tac_gap > dao_si_gap
    # Lão Tặc AI vẫn giữ độ tự tin cao dù test tệ -> "ảo tưởng gap" phải dương và lớn
    assert summary["models"][lao_tac_name]["test"]["ao_tuong_gap"] > 0.3
