import numpy as np

from .lottery_config import N_NUMBERS, PICK

BASE = N_NUMBERS + 1  # cơ số an toàn (lớn hơn giá trị số lớn nhất có thể)

def encode_tickets(tickets: np.ndarray) -> np.ndarray:
    # tickets: mảng (N, PICK) uint8, mỗi hàng là 1 vé (không cần sort trước)
    # Trả về mảng (N,) int64, mỗi phần tử là id duy nhất của vé đó (2 vé có cùng 6 số, bất kể thứ tự, sẽ luôn ra cùng 1 id)
    sorted_tickets = np.sort(tickets, axis=1).astype(np.int64)
    ids = np.zeros(sorted_tickets.shape[0], dtype=np.int64)
    for col in range(sorted_tickets.shape[1]):
        ids = ids * BASE + sorted_tickets[:, col]
    return ids

# Giải mã ngược id -> tuple 6 số đã sort tăng dần. Dùng để hiển thị, không dùng trong vòng lặp lớn (chỉ để in top combo phổ biến)
def decode_ticket(ticket_id: int) -> tuple:
    nums = []
    x = int(ticket_id)
    for _ in range(PICK):
        nums.append(x % BASE)
        x //= BASE
    return tuple(sorted(nums))
