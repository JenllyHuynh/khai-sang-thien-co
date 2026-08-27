import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import streamlit as st

st.set_page_config(page_title="Khai Sáng Thiên Cơ", page_icon="🀄", layout="wide")

st.title("Khai Sáng Thiên Cơ")
st.caption("Vạn pháp quy về Xác Suất, thiên cơ tận tại Random. Ngã độc tôn nhất đạo, dùng Code để ngộ Đạo.")

st.markdown(
    """
    Đây là **Đạo Đài** - nơi 4 trận pháp của bí kíp *Khai Sáng Thiên Cơ*
    được trình diễn trực tiếp. Chọn 1 tab bên dưới, tinh chỉnh tham số
    (nếu muốn), rồi bấm nút để mô phỏng chạy ngay trong trình duyệt.

    Đây là dự án nghiên cứu vui, minh họa các khái niệm xác suất/thống
    kê - không phải công cụ dự đoán xổ số hay tư vấn tài chính.
    """
)

tab1, tab2, tab3, tab4 = st.tabs([
    "1 Vạn Kiếp Quy Tông",
    "2 Lão Tặc AI",
    "3 Phá Giải Hỗn Mang",
    "4 Bàn Cờ Nhân Quả",
])

with tab1:
    from tran_phap_1_ui import render as render_tran_phap_1
    render_tran_phap_1()

with tab2:
    from tran_phap_2_ui import render as render_tran_phap_2
    render_tran_phap_2()

with tab3:
    from tran_phap_3_ui import render as render_tran_phap_3
    render_tran_phap_3()

with tab4:
    from tran_phap_4_ui import render as render_tran_phap_4
    render_tran_phap_4()

st.divider()
st.caption(
    "Repo: Khai Sáng Thiên Cơ - Pet Project giải trí trí tuệ, dùng Python "
    "(numpy/pandas/scipy/scikit-learn/matplotlib) + Streamlit."
)
