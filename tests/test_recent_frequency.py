import numpy as np

from paper.experiments.run_recent_frequency import select_recent
from src.tran_phap_2_overfit.data_generator import generate_draw_history


def test_recent_selection_does_not_use_current_or_future_draws():
    draws = generate_draw_history(30, seed=7)
    before = select_recent(draws, window=10, start=20)
    changed = draws.copy()
    changed[20:] = np.arange(1, 7)
    after = select_recent(changed, window=10, start=20)
    assert before.iloc[0]["selected"] == after.iloc[0]["selected"]


def test_recent_selection_breaks_frequency_ties_by_ascending_number():
    draws = np.tile(np.arange(1, 7), (12, 1))
    draws[10:] = np.arange(7, 13)
    result = select_recent(draws, window=10, start=10)
    assert result.iloc[0]["selected"] == "1 2 3 4 5 6"
    assert result.iloc[0]["overlap"] == 0
