import numpy as np

from .dynamics import CHAM_GIOI_HAN, TU_DUNG, HET_NGAN_SACH, bang_diem_suyt_trung
from .config import ThamSo


def ty_le_con_choi(so_ky: np.ndarray, t_max: int) -> np.ndarray:
    # Tỷ lệ agent còn mua vé ở kỳ thứ t (t = 1..t_max).
    return np.array([(so_ky >= t).mean() for t in range(1, t_max + 1)])


def bootstrap_trung_binh(x: np.ndarray, rng: np.random.Generator, n_boot: int = 1000,
                         ci: float = 0.95, block: int = 100):
    # Khoảng tin cậy bootstrap (percentile) cho trung bình, resample theo agent.
    # Chia khối để không cấp phát mảng n_boot x n quá lớn khi quần thể đông.
    n = x.size
    if n == 0:
        return float("nan"), float("nan")
    means = np.empty(n_boot)
    for s in range(0, n_boot, block):
        b = min(block, n_boot - s)
        means[s:s + b] = x[rng.integers(0, n, size=(b, n))].mean(axis=1)
    lo, hi = np.percentile(means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


def dem_suyt_trung(so_trung: np.ndarray, so_ky: np.ndarray, k_min: int = 3, k_max: int = 5,
                  *, bang_diem: np.ndarray | None = None) -> np.ndarray:
    # Có bảng điểm: chỉ đếm kỳ S > 0 theo cấu hình đang chạy.
    # Không có bảng điểm: giữ cách gọi cũ, đếm trùng k_min..k_max số.
    t_max = so_trung.shape[1]
    da_choi = np.arange(t_max)[None, :] < so_ky[:, None]
    gap = ((so_trung >= k_min) & (so_trung <= k_max)
           if bang_diem is None else np.asarray(bang_diem)[so_trung] > 0)
    return (gap & da_choi).sum(axis=1)


def so_sanh_cap(doi_chung: dict, nhay: dict, so_trung: np.ndarray, tham_so: ThamSo,
                rng: np.random.Generator, n_boot: int = 1000) -> dict:
    # So sánh hai bản của CÙNG agent (ghép cặp) - mọi chênh lệch là chênh lệch theo từng agent.
    d_ky = (nhay["so_ky"] - doi_chung["so_ky"]).astype(float)
    d_chi = nhay["chi"] - doi_chung["chi"]
    thua_dc = doi_chung["chi"] - doi_chung["nhan"]
    thua_nh = nhay["chi"] - nhay["nhan"]
    d_thua = thua_nh - thua_dc

    ky_dc = float(doi_chung["so_ky"].mean())
    ky_nh = float(nhay["so_ky"].mean())
    lo, hi = bootstrap_trung_binh(d_ky, rng, n_boot)
    lo_c, hi_c = bootstrap_trung_binh(d_chi, rng, n_boot)

    # Phân nhóm mô tả theo các kỳ thực sự chơi ở bản nhạy; không phải nhóm gán từ đầu.
    # Đếm đúng S > 0, kể cả khi cấu hình loại trùng 3 số khỏi tín hiệu kích thích.
    so_lan = dem_suyt_trung(
        so_trung, nhay["so_ky"], bang_diem=bang_diem_suyt_trung(tham_so.cach_gan_diem)
    )
    da_gap = so_lan > 0

    def _ly_do(ld):
        return {
            "cham_gioi_han": float((ld == CHAM_GIOI_HAN).mean()),
            "tu_dung": float((ld == TU_DUNG).mean()),
            "het_ngan_sach": float((ld == HET_NGAN_SACH).mean()),
        }

    return {
        "so_ky_tb_doi_chung": ky_dc,
        "so_ky_tb_nhay": ky_nh,
        "chenh_ky_tb": float(d_ky.mean()),
        "chenh_ky_tb_ci95": [lo, hi],
        "chenh_ky_tuong_doi_pct": float(100 * d_ky.mean() / ky_dc) if ky_dc > 0 else float("nan"),
        "ty_le_agent_choi_lau_hon": float((d_ky > 0).mean()),
        "ty_le_agent_choi_it_hon": float((d_ky < 0).mean()),  # phải = 0 do cách ghép cặp (xem README)
        "chenh_ky_lon_nhat": float(d_ky.max()),
        "chi_tb_doi_chung": float(doi_chung["chi"].mean()),
        "chi_tb_nhay": float(nhay["chi"].mean()),
        "chenh_chi_tb": float(d_chi.mean()),
        "chenh_chi_tb_ci95": [lo_c, hi_c],
        "nhan_tb_doi_chung": float(doi_chung["nhan"].mean()),
        "nhan_tb_nhay": float(nhay["nhan"].mean()),
        "thua_rong_tb_doi_chung": float(thua_dc.mean()),
        "thua_rong_tb_nhay": float(thua_nh.mean()),
        "chenh_thua_rong_tb": float(d_thua.mean()),
        "ty_le_agent_tung_gap_suyt_trung": float(da_gap.mean()),
        "chenh_ky_tb_nhom_tung_gap": float(d_ky[da_gap].mean()) if da_gap.any() else float("nan"),
        "chenh_ky_tb_nhom_khong_gap": float(d_ky[~da_gap].mean()) if (~da_gap).any() else float("nan"),
        "ly_do_dung_doi_chung": _ly_do(doi_chung["ly_do"]),
        "ly_do_dung_nhay": _ly_do(nhay["ly_do"]),
    }


def summarize(ket_qua: dict) -> dict:
    out = {
        "params": ket_qua["params"],
        "rtp_ly_thuyet": ket_qua["rtp_ly_thuyet"],
        "so_sanh": ket_qua["so_sanh"],
        "con_choi_theo_ky": ket_qua["con_choi_moc"],
    }
    if ket_qua.get("quet_beta") is not None:
        out["quet_beta"] = ket_qua["quet_beta"]
    if ket_qua.get("do_ben") is not None:
        out["do_ben"] = ket_qua["do_ben"]
    if ket_qua.get("cau_chuyen") is not None:
        out["cau_chuyen_tom_tat"] = ket_qua["cau_chuyen"]["tom_tat"]
    return out
