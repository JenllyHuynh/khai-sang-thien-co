import streamlit as st

from src.tran_phap_1_monte_carlo.aggregate import aggregate_stats
from src.tran_phap_1_monte_carlo.encode import decode_ticket
from src.tran_phap_1_monte_carlo.simulate import generate_ids_in_chunks
from src.tran_phap_1_monte_carlo.stats import cluster_stats, top_popular_combos
from src.tran_phap_1_monte_carlo.ticket_generator import generate_bias, generate_uniform
from src.tran_phap_1_monte_carlo.visualize import build_summary_figure, build_repeated_runs_figure

import numpy as np
import pandas as pd

def render() -> None:
    st.header("1 Vạn Kiếp Quy Tông - Monte Carlo Simulator")
    st.markdown(
        "> *\"Chọn số đẹp không làm tăng Thiên Mệnh, nhưng sẽ khiến tiền "
        "thưởng bị phân tán như cát bụi.\"*"
    )
    st.markdown(
        "So sánh **Phe Vô Vi** (chọn số hoàn toàn ngẫu nhiên) với **Phe "
        "Tình Cảm** (thiên vị số <= 31 và vài số hên) trong xổ số 6/55. "
        "Xác suất trúng lý thuyết của 2 phe là **như nhau**, nhưng nếu "
        "trúng thật, phe nào dễ phải **chia sẻ giải thưởng** hơn ?"
    )

    col1, col2 = st.columns(2)
    with col1:
        n_players = st.slider(
            "Số người chơi mỗi phe", min_value=50_000, max_value=3_000_000,
            value=500_000, step=50_000, key="tp1_n_players",
        )
    with col2:
        seed = st.number_input("Seed", value=42, step=1, key="tp1_seed")

    if st.button("Chạy mô phỏng", key="tp1_run"):
        with st.spinner(f"Đang sinh {n_players:,} vé mỗi phe..."):
            rng = np.random.default_rng(int(seed))
            chunk_size = min(200_000, n_players)
            ids_vo_vi = generate_ids_in_chunks(n_players, chunk_size, generate_uniform, rng, verbose=False)
            ids_tinh_cam = generate_ids_in_chunks(n_players, chunk_size, generate_bias, rng, verbose=False)

            stats_vo_vi = cluster_stats(ids_vo_vi)
            stats_tinh_cam = cluster_stats(ids_tinh_cam)

        m1, m2, m3 = st.columns(3)
        m1.metric("% trùng vé - Vô Vi", f"{stats_vo_vi['pct_players_sharing_ticket']:.4f}%")
        m2.metric("% trùng vé - Tình Cảm", f"{stats_tinh_cam['pct_players_sharing_ticket']:.4f}%")
        ty_le = (
            stats_tinh_cam["pct_players_sharing_ticket"] / stats_vo_vi["pct_players_sharing_ticket"]
            if stats_vo_vi["pct_players_sharing_ticket"] > 0 else float("inf")
        )
        m3.metric("Chênh lệch nguy cơ", f"~{ty_le:.1f}×")

        fig = build_summary_figure(stats_vo_vi, stats_tinh_cam)
        st.pyplot(fig)

        with st.expander("Top 10 tổ hợp phổ biến nhất (Phe Tình Cảm)"):
            top = top_popular_combos(stats_tinh_cam, decode_ticket, top_n=10)
            st.dataframe(top, width="stretch")

    st.divider()
    with st.expander("Nâng cao: Chạy lặp nhiều lần độc lập (Repeated Trials)"):
        st.markdown(
            "Chạy nhiều lần với seed khác nhau để xem hiện tượng có **ổn "
            "định** hay chỉ là may rủi của 1 lần chạy."
        )
        c1, c2 = st.columns(2)
        with c1:
            n_runs = st.slider("Số lần chạy", 5, 50, 15, key="tp1_n_runs")
        with c2:
            n_players_per_run = st.slider(
                "Người chơi mỗi lần chạy", 20_000, 300_000, 80_000, step=20_000, key="tp1_n_ppr",
            )

        if st.button("Chạy lặp", key="tp1_run_repeated"):
            with st.spinner(f"Đang chạy {n_runs} lần độc lập..."):
                rows = []
                for i in range(n_runs):
                    rng_i = np.random.default_rng(1000 + i)
                    chunk = min(100_000, n_players_per_run)
                    ids_a = generate_ids_in_chunks(n_players_per_run, chunk, generate_uniform, rng_i, verbose=False)
                    ids_b = generate_ids_in_chunks(n_players_per_run, chunk, generate_bias, rng_i, verbose=False)
                    sa, sb = cluster_stats(ids_a), cluster_stats(ids_b)
                    rows.append({
                        "run_index": i,
                        "n_players_per_group": n_players_per_run,
                        "vo_vi_pct_sharing": sa["pct_players_sharing_ticket"],
                        "vo_vi_unique_ratio": sa["unique_ratio"],
                        "vo_vi_max_cluster": sa["max_cluster_size"],
                        "tinh_cam_pct_sharing": sb["pct_players_sharing_ticket"],
                        "tinh_cam_unique_ratio": sb["unique_ratio"],
                        "tinh_cam_max_cluster": sb["max_cluster_size"],
                    })
                df = pd.DataFrame(rows)
                agg = aggregate_stats(df)

            fig2 = build_repeated_runs_figure(df, agg)
            st.pyplot(fig2)
