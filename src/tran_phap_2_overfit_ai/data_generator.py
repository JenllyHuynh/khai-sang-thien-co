"""
Sinh dữ liệu LỊCH SỬ QUAY SỐ giả lập cho Trận Pháp 2.

Vì không có dữ liệu quay số thật của Việt Nam (xem "Kiếp Nạn 2: Cải Trang
Dữ Liệu" trong bí kíp gốc), ta tự sinh "lịch sử 10 năm" bằng random thuần
túy — đúng bản chất của xổ số thật: các lần quay ĐỘC LẬP và ĐỒNG NHẤT,
không hề có "quy luật" ẩn nào để học. Đây chính là điểm mấu chốt của
trận pháp: cho AI học một thứ KHÔNG CÓ TÍN HIỆU THẬT, rồi xem nó "ảo
tưởng" tự tin ra sao trước khi sụp đổ trước tương lai.
"""
import numpy as np

from ..tran_phap_1_monte_carlo.lottery_config import PICK  # noqa: F401 (re-export tiện dùng)
from ..tran_phap_1_monte_carlo.ticket_generator import generate_uniform


def generate_draw_history(n_draws: int, seed: int = 42) -> np.ndarray:
    """Sinh n_draws lượt quay số độc lập, mỗi lượt PICK số phân biệt trong 1..55.

    Tái sử dụng đúng cơ chế sinh số của Trận Pháp 1 (Phe Vô Vi) — vì bản
    chất một lượt quay xổ số THẬT cũng chính là random thuần túy, không
    khác gì cách Phe Vô Vi chọn số.

    Trả về mảng (n_draws, PICK) uint8, mỗi hàng đã sort tăng dần, theo
    thứ tự thời gian (hàng 0 là lượt quay xa xưa nhất, hàng cuối là gần
    nhất).
    """
    rng = np.random.default_rng(seed)
    draws = generate_uniform(n_draws, rng)
    return np.sort(draws, axis=1)
