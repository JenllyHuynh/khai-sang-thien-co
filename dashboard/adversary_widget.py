import streamlit as st

from src.statistical_adversary import (
    critique,
    respond_to_defense,
    is_configured,
    configuration_hint,
)
from src.statistical_adversary.adversary import AdversaryError

def render_adversary(tp_key: str, tran_phap_name: str, mo_ta: str, params: dict, ket_qua: dict) -> None:
    st.divider()
    st.markdown("### Statistical Adversary")
    st.caption(
        "Một đồng nghiệp khó tính (chạy trên Gemini) sẽ soi lỗ hổng trong "
        "kết quả phía trên. Đạo hữu có dám nghênh chiến và bảo vệ lập luận "
        "của mình không?"
    )

    if not is_configured():
        st.info(configuration_hint(), icon="🔑")
        return

    history_key = f"adversary_history_{tp_key}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []
    history: list[dict] = st.session_state[history_key]

    summon_col, reset_col = st.columns([1, 1])
    with summon_col:
        summon_label = "Triệu Hồi Đối Thủ" if not history else "Triệu Hồi Lại (bỏ hội thoại cũ)"
        summon = st.button(summon_label, key=f"adversary_summon_{tp_key}")
    with reset_col:
        if history and st.button("Xoá hội thoại", key=f"adversary_clear_{tp_key}"):
            st.session_state[history_key] = []
            st.rerun()

    if summon:
        with st.spinner("Đối thủ đang soi bài của đạo hữu..."):
            try:
                text = critique(tran_phap_name, mo_ta, params, ket_qua)
                st.session_state[history_key] = [{"role": "adversary", "text": text}]
            except AdversaryError as exc:
                st.error(str(exc))
        st.rerun()

    history = st.session_state[history_key]
    for turn in history:
        is_adversary = turn["role"] == "adversary"
        with st.chat_message("assistant" if is_adversary else "user", avatar="👹" if is_adversary else "🧑"):
            st.markdown(turn["text"])

    if history:
        defense = st.text_area(
            "Giải trình / phản biện lại đối thủ:",
            key=f"adversary_defense_{tp_key}",
            height=100,
            placeholder="Đạo hữu bảo vệ kết luận của mình thế nào? (vd: giải thích vì sao cỡ mẫu / tham số đã đủ, hoặc thừa nhận hạn chế và đề xuất hướng khắc phục)",
        )
        send = st.button("Gửi phản biện", key=f"adversary_reply_{tp_key}")
        if send:
            if not defense.strip():
                st.warning("Đạo hữu chưa nhập gì để gửi cả.")
            else:
                with st.spinner("Đối thủ đang cân nhắc lập luận của đạo hữu..."):
                    try:
                        reply = respond_to_defense(history, defense.strip())
                        st.session_state[history_key].append({"role": "user", "text": defense.strip()})
                        st.session_state[history_key].append({"role": "adversary", "text": reply})
                    except AdversaryError as exc:
                        st.error(str(exc))
                st.rerun()
