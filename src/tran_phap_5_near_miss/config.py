from dataclasses import dataclass, field

# Luật chơi (giữ nhất quán với Trận Pháp 1)
N_NUMBERS = 55  # Dải số 1..55
PICK = 6        # Mỗi vé / mỗi kỳ quay 6 số

# Giá vé và bảng thưởng: GIẢ ĐỊNH MINH HỌA, cỡ gần một loại xổ số 6/55 phổ biến
# nhưng KHÔNG phải bảng thưởng chính thức. Jackpot cố định (không lăn, không chia
# giải), không có giải phụ (số bonus). Ai cần bám luật thật thì tự thay bảng này.
GIA_VE = 10_000  # đồng / vé

# số trùng -> tiền thưởng (đồng). Trùng 0-2 số: không có thưởng.
BANG_THUONG_MAC_DINH = {
    3: 50_000,
    4: 500_000,
    5: 40_000_000,
    6: 30_000_000_000,
}

# Các cách gán điểm "suýt trúng" S_t theo số trùng.
# Trùng 6 số là jackpot (thắng thật) nên KHÔNG tính là suýt trúng. Trùng 0-2: S=0.
CACH_GAN_DIEM = {
    "tuyen_tinh": {3: 1 / 3, 4: 2 / 3, 5: 1.0},    # gần jackpot tăng đều
    "nhan_manh_gan_nhat": {3: 0.1, 4: 0.4, 5: 1.0},  # chỉ khi sát nút mới "đã"
    "tu_3_so_nhu_nhau": {3: 1.0, 4: 1.0, 5: 1.0},    # trùng >=3 là như nhau
    "chi_4_5_so": {3: 0.0, 4: 0.5, 5: 1.0},          # 3 số chưa đủ "gần"
}


@dataclass(frozen=True)
class ThamSo:
    # Động lực hứng thú: H' = rho*H + alpha*R + beta*S
    rho: float = 0.985
    alpha: float = 0.06
    beta: float = 0.0            # 0 = đối chứng; > 0 = nhạy với suýt trúng
    r_cap: float = 5.0           # chặn |R_t| để tiền thưởng lớn không áp đảo

    cach_gan_diem: str = "tuyen_tinh"

    # Quần thể agent
    h0_mean: float = 4.5
    h0_std: float = 0.8
    ngan_sach_median_ve: float = 40.0   # ngân sách trung vị tính bằng số vé
    ngan_sach_sigma: float = 0.8        # độ phân tán lognormal

    # Vòng chơi
    max_ky: int = 300
    gia_ve: int = GIA_VE
    bang_thuong: dict = field(default_factory=lambda: dict(BANG_THUONG_MAC_DINH))
