from dataclasses import dataclass

# Luật chơi
N_NUMBERS = 55  # Dải số 1..55
PICK = 6        # Chọn 6 số mỗi vé

# Đặc điểm "Phe Tình Cảm" (Tâm Lý Học)
# Dân gian hay chọn số theo ngày sinh (<=31) và vài con số được cho là "hên"
BIRTHDAY_MAX = 31

# Số hên trong bí kíp gốc có "67" nhưng vượt quá dải 1..55 nên bị loại,
# giữ lại phần hợp lệ.
LUCKY_NUMBERS = frozenset({1, 9, 11, 20, 36})

# Trọng số dùng để mô phỏng hành vi chọn số 'thiên vị' của Phe Tình Cảm
@dataclass(frozen=True)
class BiasConfig:
    # Trọng số càng cao thì số đó càng dễ được chọn (weighted sampling without replacement, xem ticket_generator.py).
    base_weight: float = 1.0       # số ngoài dải sinh nhật, không phải số hên
    birthday_weight: float = 3.0   # số trong dải 1..31 (ngày sinh)
    lucky_weight: float = 6.0      # số hên đặc biệt (ưu tiên cao nhất)
