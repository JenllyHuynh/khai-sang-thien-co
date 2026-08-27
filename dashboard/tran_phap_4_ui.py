import streamlit as st

from src.tran_phap_4_hoa_phuc.simulate import run_simulation
from src.tran_phap_4_hoa_phuc.stats import summarize
from src.tran_phap_4_hoa_phuc.visualize import build_full_figure

def render() -> None:
    st.header("4 Bàn Cờ Nhân Quả - Họa Phúc Song Hành")
    st.markdown(
        "> *\"Trong họa có phúc, trong phúc có họa. Cả hai đều là... "
        "label của con người.\"*"
    )
    st.markdown(
        "Kiểm định thống kê: **Trúng số (A)** và **Gặp họa (B)** được mô "
        "phỏng ĐỘC LẬP tuyệt đối. Chi-square test có phát hiện được "
        "\"lời nguyền\" nào không - hay tất cả chỉ là ngụy biện tường thuật ?"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n_people = st.slider("Kích thước quần thể", 200_000, 3_000_000, 1_000_000, step=200_000, key="tp4_n_people")
    with col2:
        p_win = st.select_slider(
            "Xác suất 'trúng số' (phóng đại để đủ mẫu)",
            options=[0.0002, 0.0005, 0.001, 0.002, 0.005], value=0.001, key="tp4_p_win",
        )
    with col3:
        p_hoa = st.slider("Xác suất nền 'gặp họa'", 0.01, 0.2, 0.05, step=0.01, key="tp4_p_hoa")

    n_trials_null = st.slider("Số lần chạy độc lập (phân phối null)", 30, 200, 100, step=10, key="tp4_n_trials")

    if st.button("Chạy mô phỏng", key="tp4_run"):
        with st.spinner(f"Đang dựng quần thể {n_people:,} người + {n_trials_null} lần chạy độc lập..."):
            result = run_simulation(
                n_people=n_people, p_win=p_win, p_hoa_baseline=p_hoa,
                n_trials_null=n_trials_null, seed=42,
            )
            summary = summarize(result)

        m1, m2, m3 = st.columns(3)
        m1.metric("P(Họa | Trúng)", f"{summary['p_hoa_given_trung']*100:.3f}%")
        m2.metric("P(Họa | Không Trúng)", f"{summary['p_hoa_given_khong_trung']*100:.3f}%")
        y_nghia = "CÓ ý nghĩa" if summary["co_y_nghia_thong_ke"] else "KHÔNG có ý nghĩa"
        m3.metric("Chi² p-value", f"{summary['p_value']:.3f}", help=f"{y_nghia} thống kê ở mức 5%")

        fig = build_full_figure(result["table"], result["independence_result"], result["null_diffs"])
        st.pyplot(fig)

        st.caption(
            f"Odds Ratio = {summary['odds_ratio']:.3f} (≈1.0 = không liên hệ). "
            f"Qua {summary['params']['n_trials_null']} lần chạy độc lập, chênh lệch dao động quanh "
            f"{summary['null_distribution_mean']:+.3f} điểm % (độ lệch chuẩn {summary['null_distribution_std']:.3f})."
        )
