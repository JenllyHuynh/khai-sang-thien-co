import streamlit as st

from src.tran_phap_2_overfit_ai.simulate import run_simulation
from src.tran_phap_2_overfit_ai.stats import summarize
from src.tran_phap_2_overfit_ai.visualize import build_overfitting_figure

def render() -> None:
    st.header("2 Lão Tặc AI - Kẻ Học Vẹt (Overfitted Prophet)")
    st.markdown(
        "> *\"AI tưởng mình thông thiên, nhưng trước Random chỉ là một "
        "thằng học vẹt.\"*"
    )
    st.markdown(
        "Huấn luyện 2 model trên lịch sử quay số giả lập: **Lão Tặc AI** "
        "được cố tình cho \"xem đáp án\" (số thứ tự lần quay) để nó học "
        "thuộc lòng quá khứ; **Đạo Sĩ Khiêm Tốn** không có đặc quyền đó. "
        "Cả hai bị kiểm tra trên dữ liệu **tương lai chưa từng thấy**."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n_draws = st.slider("Số lần quay lịch sử", 300, 5000, 1560, step=100, key="tp2_n_draws")
    with col2:
        window = st.slider("Cửa sổ tần suất gần đây", 10, 100, 52, key="tp2_window")
    with col3:
        test_frac = st.slider("Tỷ lệ dữ liệu 'tương lai'", 0.05, 0.3, 0.1, step=0.05, key="tp2_test_frac")

    if st.button("Chạy mô phỏng", key="tp2_run"):
        with st.spinner("Đang huấn luyện Lão Tặc AI và Đạo Sĩ Khiêm Tốn..."):
            result = run_simulation(n_draws=n_draws, window=window, test_frac=test_frac, seed=42)
            summary = summarize(result)

        lao_tac_name = [n for n in summary["models"] if "Học Vẹt" in n][0]
        dao_si_name = [n for n in summary["models"] if "Baseline" in n][0]
        lt, ds = summary["models"][lao_tac_name], summary["models"][dao_si_name]

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Lão Tặc AI - số trúng (train -> test)",
            f"{lt['test']['avg_matches']:.2f}/6",
            delta=f"từ {lt['train']['avg_matches']:.2f}/6 lúc train",
            delta_color="off",
        )
        m2.metric(
            "Độ tự tin Lão Tặc AI (test)",
            f"{lt['test']['avg_confidence']*100:.1f}%",
        )
        m3.metric(
            "Khoảng cách ảo tưởng",
            f"{lt['test']['ao_tuong_gap']*100:+.1f} điểm %",
            help="Độ tự tin AI tự nhận − tỷ lệ THỰC SỰ đúng. Càng lớn càng ảo tưởng.",
        )

        fig = build_overfitting_figure(summary)
        st.pyplot(fig)

        st.caption(
            f"Đoán mò lý thuyết: {summary['theoretical_random_matches']:.3f}/6 số trúng. "
            f"Đạo Sĩ Khiêm Tốn: {ds['test']['avg_matches']:.2f}/6 trên tương lai "
            f"(khoảng cách ảo tưởng chỉ {ds['test']['ao_tuong_gap']*100:+.1f} điểm % - trung thực hơn hẳn)."
        )
