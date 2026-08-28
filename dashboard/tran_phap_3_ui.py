import streamlit as st

from dashboard.adversary_widget import render_adversary
from src.tran_phap_3_chaos_prng.simulate import run_simulation
from src.tran_phap_3_chaos_prng.visualize import build_chaos_figure, build_prng_structure_figure


def render() -> None:
    st.header("3 Phá Giải Hỗn Mang - Chaos vs. PRNG")
    st.markdown(
        "> *\"Hỗn mang có quy luật, nhưng Xổ Số là bậc thầy của vô "
        "thường.\"*"
    )
    st.markdown(
        "**Phần 1:** hệ Lorenz - hỗn loạn TẤT ĐỊNH, trông ngẫu nhiên "
        "nhưng cực nhạy với điều kiện ban đầu (hiệu ứng cánh bướm). "
        "**Phần 2:** bắt mạch PRNG yếu (RANDU) chỉ với 1 số quan sát, "
        "đối chứng với CSPRNG mạnh (PCG64 - bộ sinh số dùng trong cả bí kíp)."
    )

    col1, col2 = st.columns(2)
    with col1:
        n_steps_attractor = st.slider(
            "Số bước quỹ đạo Lorenz", 1000, 20000, 6000, step=1000, key="tp3_n_steps",
        )
    with col2:
        n_prng_samples = st.slider(
            "Số mẫu PRNG (mỗi loại)", 1000, 20000, 5000, step=1000, key="tp3_n_prng",
        )

    if st.button("Chạy mô phỏng", key="tp3_run"):
        with st.spinner("Đang tích phân hệ Lorenz và sinh chuỗi PRNG..."):
            result = run_simulation(
                n_steps_attractor=n_steps_attractor,
                dt=0.01,
                n_steps_butterfly=min(3000, n_steps_attractor),
                n_prng_samples=n_prng_samples,
                seed=42,
            )

        st.session_state["tp3_result"] = {
            "result": result,
            "n_steps_attractor": n_steps_attractor,
            "n_prng_samples": n_prng_samples,
        }

    saved = st.session_state.get("tp3_result")
    if saved is not None:
        result = saved["result"]
        lyap = result["lyapunov"]
        crack_weak = result["crack_weak_result"]
        crack_strong = result["naive_crack_on_strong"]

        m1, m2, m3 = st.columns(3)
        if lyap.get("doubling_time"):
            m1.metric("Thời gian nhân đôi sai số", f"~{lyap['doubling_time']:.2f} đơn vị")
        m2.metric("Bắt mạch LCG yếu", "THÀNH CÔNG" if crack_weak["cracked"] else "thất bại",
                   help=f"Sai số tối đa: {crack_weak['max_abs_error']:.2e}")
        m3.metric("Thử bắt mạch CSPRNG mạnh", "thất bại" if not crack_strong["cracked"] else "(?!)",
                   help=f"Sai số tối đa: {crack_strong['max_abs_error']:.2e}")

        st.subheader("Phần 1: Hỗn Mang Có Quy Luật")
        fig1 = build_chaos_figure(result["trajectory"], result["butterfly_times"], result["butterfly_distance"], lyap)
        st.pyplot(fig1)

        st.subheader("Phần 2: Bắt Mạch Yêu Quái")
        fig2 = build_prng_structure_figure(result["weak_lcg_samples"], result["strong_csprng_samples"])
        st.pyplot(fig2)

        render_adversary(
            tp_key="tp3",
            tran_phap_name="Trận Pháp 3 - Phá Giải Hỗn Mang (Chaos vs. PRNG)",
            mo_ta=(
                "Phần 1: tích phân số hệ Lorenz (deterministic chaos), đo hệ "
                "số Lyapunov / thời gian nhân đôi sai số giữa 2 quỹ đạo có "
                "điều kiện ban đầu gần nhau (hiệu ứng cánh bướm). Phần 2: thử "
                "'bắt mạch' (dự đoán số kế tiếp) một PRNG yếu (LCG kiểu RANDU) "
                "và một CSPRNG mạnh (PCG64) chỉ từ các số quan sát được, để "
                "đối chiếu hỗn loạn tất định với ngẫu nhiên mật mã học."
            ),
            params={
                "n_buoc_quy_dao_lorenz": saved["n_steps_attractor"],
                "dt": 0.01,
                "n_buoc_butterfly": min(3000, saved["n_steps_attractor"]),
                "n_mau_prng_moi_loai": saved["n_prng_samples"],
                "seed": 42,
            },
            ket_qua={
                "lyapunov": lyap,
                "bat_mach_lcg_yeu": {
                    "cracked": crack_weak["cracked"],
                    "max_abs_error": crack_weak["max_abs_error"],
                },
                "thu_bat_mach_csprng_manh": {
                    "cracked": crack_strong["cracked"],
                    "max_abs_error": crack_strong["max_abs_error"],
                },
            },
        )
