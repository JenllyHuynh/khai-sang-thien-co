from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from ..tran_phap_1_monte_carlo.lottery_config import PICK

# Lão Tặc AI được "cho xem đáp án" qua draw_index + number
FEATURES_OVERFIT = ["draw_index", "number", "freq_last_window", "overall_freq", "gap_since_last"]
# Đạo Sĩ Khiêm Tốn chỉ có đặc trưng thống kê thuần túy, không có định danh
FEATURES_BASELINE = ["freq_last_window", "overall_freq", "gap_since_last"]

@dataclass
class TrainedModel:
    name: str
    model: DecisionTreeClassifier
    feature_cols: list

# Huấn luyện 'Lão Tặc AI' - cố tình để nó overfit/học vẹt
def train_lao_tac_ai(df_train: pd.DataFrame) -> TrainedModel:
    model = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, random_state=42)
    model.fit(df_train[FEATURES_OVERFIT], df_train["label"])
    return TrainedModel(name="Lão Tặc AI (Học Vẹt)", model=model, feature_cols=FEATURES_OVERFIT)

#  Huấn luyện 'Đạo Sĩ Khiêm Tốn' - model được kiểm soát overfitting đàng hoàng
def train_dao_si_khiem_ton(df_train: pd.DataFrame) -> TrainedModel:
    model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=50, random_state=42)
    model.fit(df_train[FEATURES_BASELINE], df_train["label"])
    return TrainedModel(name="Đạo Sĩ Khiêm Tốn (Baseline)", model=model, feature_cols=FEATURES_BASELINE)

def predict_top6_per_draw(trained: TrainedModel, df: pd.DataFrame) -> pd.DataFrame:
    proba = trained.model.predict_proba(df[trained.feature_cols])
    classes = trained.model.classes_
    if len(classes) == 2:
        p1 = proba[:, list(classes).index(1)]
    else:
        # Trường hợp hiếm: model chỉ thấy 1 lớp trong lúc train
        p1 = proba[:, 0] if classes[0] == 1 else np.zeros(len(df))

    work = df[["draw_index", "number", "label"]].copy()
    work["p1"] = p1

    rows = []
    for draw_index, g in work.groupby("draw_index", sort=False):
        top = g.nlargest(PICK, "p1")
        rows.append({
            "draw_index": draw_index,
            "so_khop": int(top["label"].sum()),
            "do_tin_binh_quan": float(top["p1"].mean()),
        })
    return pd.DataFrame(rows)
