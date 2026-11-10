import numpy as np

from .config import ThamSo
from .draws import dung_ky_quay_co_k_trung
from .dynamics import (
    bang_diem_suyt_trung, bang_thuong_theo_so_trung, chay_mot_ban, TU_DUNG, HET_NGAN_SACH,
)

TEN_LY_DO = {0: "chạm giới hạn số kỳ", TU_DUNG: "tự dừng", HET_NGAN_SACH: "hết ngân sách"}


def _chon_dai_dien(hat, dc, nh, tham_so) -> int | None:
    d = nh["so_ky"] - dc["so_ky"]
    ung_vien = np.flatnonzero(d > 0)
    if ung_vien.size == 0:
        return None
    trung_vi = np.median(d[ung_vien])
    return int(ung_vien[np.argmin(np.abs(d[ung_vien] - trung_vi))])


def dung_cau_chuyen(hat, tham_so: ThamSo, beta: float, dc, nh, seed: int) -> dict | None:
    i = _chon_dai_dien(hat, dc, nh, tham_so)
    if i is None:
        return None

    sl = slice(i, i + 1)
    pop = hat["pop"]
    args = (hat["so_trung"][sl], hat["u"][sl], pop["ngan_sach"][sl], pop["h0"][sl], tham_so)
    v_dc = chay_mot_ban(*args, beta=0.0, giu_vet=True)
    v_nh = chay_mot_ban(*args, beta=beta, giu_vet=True)

    n_dc, n_nh = int(v_dc["so_ky"][0]), int(v_nh["so_ky"][0])
    thuong = bang_thuong_theo_so_trung(tham_so)
    diem = bang_diem_suyt_trung(tham_so.cach_gan_diem)
    ve = pop["bo_so"][i]
    rng = np.random.default_rng(seed + 99)

    cac_ky = []
    for t in range(n_nh):
        k = int(hat["so_trung"][i, t])
        quay = dung_ky_quay_co_k_trung(rng, ve, k)
        cac_ky.append({
            "ky": t + 1,
            "quay": [int(x) for x in quay],
            "so_trung": k,
            "thuong": float(thuong[k]),
            "diem_suyt_trung": float(diem[k]),
            "h_doi_chung": None if np.isnan(v_dc["vet_h"][0, t]) else float(v_dc["vet_h"][0, t]),
            "h_nhay": float(v_nh["vet_h"][0, t]),
        })

    ky_gap = next((c for c in cac_ky if c["diem_suyt_trung"] > 0), None)
    h_dau = float(pop["h0"][i])
    dong = [
        f"Agent #{i}: ngân sách {int(pop['ngan_sach'][i]):,}đ ({int(pop['ngan_sach'][i] // tham_so.gia_ve)} vé), "
        f"bộ số {' '.join(f'{x:02d}' for x in ve)}, hứng thú ban đầu H0 = {h_dau:.2f}.",
    ]
    if ky_gap is not None:
        truoc = [c for c in cac_ky if c["ky"] < ky_gap["ky"]]
        thua_truoc = sum(1 for c in truoc if c["thuong"] == 0)
        h_truoc = truoc[-1]["h_nhay"] if truoc else h_dau
        dong.append(
            f"Kỳ 1-{ky_gap['ky'] - 1}: không có kỳ nào sát nút ({thua_truoc} kỳ trắng thưởng); "
            f"hứng thú giảm từ {h_dau:.2f} xuống {h_truoc:.2f}."
            if truoc else "Kỳ 1 đã gặp suýt trúng ngay."
        )
        dong.append(
            f"Kỳ {ky_gap['ky']}: quay {' '.join(f'{x:02d}' for x in ky_gap['quay'])}, "
            f"trùng {ky_gap['so_trung']} số (nhận {int(ky_gap['thuong']):,}đ). "
            f"Bản đối chứng: H {h_truoc:.2f} -> {ky_gap['h_doi_chung']:.2f}. "
            f"Bản nhạy suýt trúng: H {h_truoc:.2f} -> {ky_gap['h_nhay']:.2f}."
        )
    dong.append(
        f"Bản đối chứng dừng sau {n_dc} kỳ ({TEN_LY_DO[int(v_dc['ly_do'][0])]}); "
        f"bản nhạy suýt trúng dừng sau {n_nh} kỳ ({TEN_LY_DO[int(v_nh['ly_do'][0])]}). "
        f"Cùng ngân sách, cùng bộ số, cùng chuỗi quay, cùng số ngẫu nhiên quyết định."
    )
    dong.append(
        f"Lưu ý: agent này được chọn vì chênh lệch ({n_nh - n_dc} kỳ) gần trung vị của nhóm bị kéo dài, "
        f"không phải vì ấn tượng nhất; và nhóm đó chỉ là một phần nhỏ trong quần thể."
    )

    return {
        "agent": i,
        "bo_so": [int(x) for x in ve],
        "cac_ky": cac_ky,
        "ky_dung_doi_chung": n_dc,
        "ky_dung_nhay": n_nh,
        "tom_tat": " ".join(dong),
        "cac_dong": dong,
    }
