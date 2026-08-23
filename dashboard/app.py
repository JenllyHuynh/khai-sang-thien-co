"""
Đạo Đài (Dashboard) - Khai Sáng Thiên Cơ
Nơi các đạo hữu vào chiêm ngưỡng chân lý xác suất và cười ha hả.

Chạy bằng: streamlit run dashboard/app.py
"""
import streamlit as st

st.set_page_config(page_title="Khai Sáng Thiên Cơ", page_icon="🀄", layout="wide")

st.title("🀄 Khai Sáng Thiên Cơ - Đạo Đài")
st.caption("Vạn pháp quy về Xác Suất, thiên cơ tận tại Random.")

st.markdown(
    """
    Chào đạo hữu! Đây là Đạo Đài trung tâm, nơi hội tụ Tứ Đại Trận Pháp:

    1. **Vạn Kiếp Quy Tông** — Monte Carlo Simulator
    2. **Lão Tặc AI** — Overfitted Prophet
    3. **Phá Giải Hỗn Mang** — Chaos vs. PRNG
    4. **Bàn Cờ Nhân Quả** — Quantifying Luck

    Các trận pháp hiện đang được luyện chế, tiến vào từng module trong
    `src/` để xem chi tiết. Trang này sẽ được ghép nối thành giao diện
    hoàn chỉnh khi các trận pháp hoàn thiện.
    """
)

st.info("🚧 Đạo Đài đang xây dựng — các trận pháp sẽ lần lượt xuất hiện tại đây.")
