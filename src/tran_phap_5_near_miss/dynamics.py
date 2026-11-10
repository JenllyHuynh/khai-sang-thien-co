import numpy as np

from .config import ThamSo, CACH_GAN_DIEM

# Lý do dừng
CHAM_GIOI_HAN = 0   # chạm giới hạn số kỳ
TU_DUNG = 1         # agent tự dừng
HET_NGAN_SACH = 2   # không đủ tiền mua vé kỳ sau


def bang_thuong_theo_so_trung(tham_so: ThamSo) -> np.ndarray:
    arr = np.zeros(7)
    for k, v in tham_so.bang_thuong.items():
        arr[k] = v
    return arr


def bang_diem_suyt_trung(ten_cach: str) -> np.ndarray:
    arr = np.zeros(7)
    for k, v in CACH_GAN_DIEM[ten_cach].items():
        arr[k] = v
    return arr


def chuan_hoa_tien(net_theo_ve: np.ndarray, r_cap: float) -> np.ndarray:
    # R_t: lãi/lỗ tính theo đơn vị giá vé, nén kiểu log (nhạy giảm dần), chặn ở ±r_cap.
    # Thua trọn vé (net = -1 vé) cho R = -ln 2 ~ -0.69.
    r = np.sign(net_theo_ve) * np.log1p(np.abs(net_theo_ve))
    return np.clip(r, -r_cap, r_cap)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def chay_mot_ban(
    so_trung: np.ndarray,      # (N, T) số trùng mỗi kỳ, CHUNG cho hai bản
    u: np.ndarray,             # (N, T) số ngẫu nhiên dùng để quyết định, CHUNG cho hai bản
    ngan_sach: np.ndarray,     # (N,)
    h0: np.ndarray,            # (N,)
    tham_so: ThamSo,
    beta: float,
    giu_vet: bool = False,
) -> dict:
    n, t_max = so_trung.shape
    gia = tham_so.gia_ve
    thuong = bang_thuong_theo_so_trung(tham_so)
    diem = bang_diem_suyt_trung(tham_so.cach_gan_diem)

    hoat_dong = ngan_sach >= gia
    tien = ngan_sach.astype(np.float64).copy()
    h = h0.astype(np.float64).copy()
    so_ky = np.zeros(n, dtype=np.int64)
    chi = np.zeros(n)
    nhan = np.zeros(n)
    ly_do = np.full(n, CHAM_GIOI_HAN, dtype=np.int8)
    ly_do[~hoat_dong] = HET_NGAN_SACH

    vet_h = np.full((n, t_max), np.nan) if giu_vet else None

    for t in range(t_max):
        choi = hoat_dong
        if not choi.any():
            break
        k = so_trung[:, t]
        tra = np.where(choi, thuong[k], 0.0)
        tien = tien - np.where(choi, gia, 0.0) + tra
        chi += np.where(choi, gia, 0.0)
        nhan += tra
        so_ky += choi

        net_ve = (tra - np.where(choi, gia, 0.0)) / gia
        r = chuan_hoa_tien(net_ve, tham_so.r_cap)
        s = diem[k]
        h = np.where(choi, tham_so.rho * h + tham_so.alpha * r + beta * s, h)
        if giu_vet:
            vet_h[:, t] = np.where(choi, h, np.nan)

        het_tien = choi & (tien < gia)
        bo_cuoc = choi & ~het_tien & (u[:, t] >= sigmoid(h))
        ly_do[het_tien] = HET_NGAN_SACH
        ly_do[bo_cuoc] = TU_DUNG
        hoat_dong = choi & ~het_tien & ~bo_cuoc

    return {
        "so_ky": so_ky, "chi": chi, "nhan": nhan, "ly_do": ly_do, "h_cuoi": h,
        "vet_h": vet_h,
    }
