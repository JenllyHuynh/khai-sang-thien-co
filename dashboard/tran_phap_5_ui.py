from dataclasses import replace

import streamlit as st

from dashboard.adversary_widget import render_adversary
from src.tran_phap_5_near_miss.config import CACH_GAN_DIEM, ThamSo
from src.tran_phap_5_near_miss.simulate import chay_thi_nghiem
from src.tran_phap_5_near_miss.stats import summarize
from src.tran_phap_5_near_miss.visualize import build_main_figure, build_sweep_figure

BETAS_QUET = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
RHOS_DO_BEN = [0.95, 0.97, 0.985, 0.995]


def render() -> None:
    st.header("5 Tâm Ma Suýt Trúng - Behavioral Agent Simulation")
    st.markdown(
        "> *\"Cảm giác suýt trúng có thể khiến một người ở lại cuộc chơi lâu hơn như thế nào?\"*"
    )
    st.markdown(
        "Mỗi agent được chạy **hai bản** với cùng ngân sách, cùng bộ số, cùng chuỗi quay và cùng "
        "số ngẫu nhiên quyết định. Khác biệt duy nhất là β: bản đối chứng có β=0, bản nhạy suýt "
        "trúng có β>0. Suýt trúng **không** làm xác suất kỳ sau tăng lên; nó chỉ cộng hứng thú cho agent."
    )
    st.warning(
        "Các tham số (ρ, α, β, cách gán điểm, ngân sách...) là giả định để khảo sát, chưa đo từ người thật. "
        "Chiều của hiệu ứng được lập trình sẵn; thứ đáng xem là độ lớn và độ nhạy với giả định.",
        icon="⚠️",
    )
    st.caption(
        "Mỗi agent có chuỗi kết quả riêng, độc lập với agent khác; hai bản của cùng agent dùng chung chuỗi đó. "
        "Mô hình không dùng một kỳ quay chung cho toàn quần thể."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        n_agents = st.slider("Số agent", 2_000, 30_000, 10_000, step=2_000, key="tp5_n")
        beta = st.slider("β (độ nhạy suýt trúng của bản nhạy)", 0.0, 3.0, 1.0, step=0.25, key="tp5_beta")
    with c2:
        rho = st.slider("ρ (phần hứng thú giữ lại)", 0.90, 0.995, 0.985, step=0.005, key="tp5_rho")
        alpha = st.slider("α (độ nhạy với kết quả tiền bạc)", 0.0, 0.3, 0.06, step=0.01, key="tp5_alpha")
    with c3:
        cach = st.selectbox("Cách gán điểm suýt trúng S", list(CACH_GAN_DIEM), index=0, key="tp5_cach")
        quet = st.checkbox("Kèm quét β và kiểm tra độ bền", value=True, key="tp5_quet")

    with st.expander("Bảng điểm suýt trúng S đang chọn"):
        st.write({f"trùng {k} số": round(v, 3) for k, v in CACH_GAN_DIEM[cach].items()})

    if st.button("Chạy mô phỏng", key="tp5_run"):
        tham_so = replace(ThamSo(), rho=rho, alpha=alpha, cach_gan_diem=cach)
        with st.spinner(f"Đang chạy {n_agents:,} agent x 2 bản..."):
            kq = chay_thi_nghiem(
                n_agents=n_agents, tham_so=tham_so, beta=beta, seed=42,
                betas_quet=BETAS_QUET if quet else None,
                rhos_do_ben=RHOS_DO_BEN if quet else None,
            )
        # bỏ các mảng lớn (_hat, _dc, _nh) trước khi lưu vào session
        st.session_state["tp5_result"] = {k: v for k, v in kq.items() if not k.startswith("_")}

    kq = st.session_state.get("tp5_result")
    if kq is None:
        return

    summary = summarize(kq)
    ss = summary["so_sanh"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Số kỳ TB (đối chứng → nhạy)", f"{ss['so_ky_tb_doi_chung']:.1f} → {ss['so_ky_tb_nhay']:.1f}",
              delta=f"{ss['chenh_ky_tuong_doi_pct']:+.2f}%")
    m2.metric("Chênh số kỳ (CI95)", f"{ss['chenh_ky_tb']:+.2f}",
              help=f"Bootstrap theo agent: {ss['chenh_ky_tb_ci95'][0]:+.2f} đến {ss['chenh_ky_tb_ci95'][1]:+.2f}")
    m3.metric("Agent chơi lâu hơn", f"{ss['ty_le_agent_choi_lau_hon']*100:.1f}%")
    m4.metric("Chi thêm TB / agent", f"{ss['chenh_chi_tb']:+,.0f}đ")

    st.pyplot(build_main_figure(kq))

    if kq["cau_chuyen"] is not None:
        st.markdown("**Câu chuyện một agent**")
        for dong in kq["cau_chuyen"]["cac_dong"]:
            st.markdown(f"- {dong}")

    st.caption(
        f"{ss['ty_le_agent_tung_gap_suyt_trung']*100:.1f}% agent từng gặp tín hiệu S > 0 theo cấu hình "
        f"trong các kỳ thực sự chơi ở bản nhạy; nhóm này chơi thêm TB {ss['chenh_ky_tb_nhom_tung_gap']:.2f} kỳ. "
        "Đây là phân nhóm mô tả sau mô phỏng, không phải nhóm đối chứng độc lập. "
        f"Tiền thưởng kỳ vọng / giá vé theo bảng giả định: {summary['rtp_ly_thuyet']*100:.1f}%."
    )

    fig2 = build_sweep_figure(kq["quet_beta"], kq["do_ben"])
    if fig2 is not None:
        st.markdown("**Nhạy đến đâu với giả định?**")
        st.pyplot(fig2)

    render_adversary(
        tp_key="tp5",
        tran_phap_name="Trận Pháp 5 - Tâm Ma Suýt Trúng (Behavioral Agent Simulation)",
        mo_ta=(
            "Mô phỏng agent mua 1 vé/kỳ cho xổ số 6/55. Hứng thú H' = rho*H + alpha*R + beta*S, "
            "xác suất tiếp tục = sigmoid(H). R là kết quả tiền bạc chuẩn hóa, S là điểm 'suýt trúng' "
            "theo bảng điểm cấu hình cho số trùng 3-5; chỉ S>0 mới tính là tín hiệu suýt trúng. "
            "Các agent có chuỗi kết quả độc lập. Mỗi agent chạy hai bản ghép cặp (đối chứng beta=0 và nhạy beta>0) với "
            "cùng ngân sách, bộ số, chuỗi quay và số ngẫu nhiên quyết định, nên bản nhạy không bao giờ "
            "chơi ít hơn bản đối chứng. Tham số là giả định, chưa đo từ người thật."
        ),
        params=summary["params"],
        ket_qua={k: v for k, v in summary.items() if k != "params"},
    )
