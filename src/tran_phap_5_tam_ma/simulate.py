from dataclasses import replace, asdict

import numpy as np

from .agents import tao_quan_the
from .config import ThamSo, CACH_GAN_DIEM
from .draws import sample_so_trung, pmf_so_trung
from .dynamics import bang_thuong_theo_so_trung
from .dynamics import chay_mot_ban
from .stats import so_sanh_cap, ty_le_con_choi
from .story import dung_cau_chuyen

MOC_KY = (5, 10, 20, 30, 50, 100)


def ty_le_hoan_tra_ly_thuyet(tham_so: ThamSo) -> float:
    return float((pmf_so_trung() * bang_thuong_theo_so_trung(tham_so)).sum() / tham_so.gia_ve)


def tao_hat_giong(n_agents: int, tham_so: ThamSo, seed: int) -> dict:
    """Mọi thứ NGẪU NHIÊN được sinh một lần và dùng chung cho cả hai bản của mỗi agent:
    ngân sách, bộ số, H0, chuỗi số trùng từng kỳ, và số ngẫu nhiên dùng để quyết định.
    Các agent khác nhau có chuỗi kết quả độc lập; không có kỳ quay chung cho quần thể.
    """
    s_agent, s_quay, s_quyet = np.random.SeedSequence(seed).spawn(3)
    pop = tao_quan_the(n_agents, tham_so, np.random.default_rng(s_agent))
    so_trung = sample_so_trung(np.random.default_rng(s_quay), (n_agents, tham_so.max_ky))
    u = np.random.default_rng(s_quyet).random((n_agents, tham_so.max_ky))
    return {"pop": pop, "so_trung": so_trung, "u": u}


def chay_cap(hat: dict, tham_so: ThamSo, beta: float, giu_vet: bool = False) -> tuple[dict, dict]:
    pop = hat["pop"]
    dc = chay_mot_ban(hat["so_trung"], hat["u"], pop["ngan_sach"], pop["h0"], tham_so, beta=0.0, giu_vet=giu_vet)
    nh = chay_mot_ban(hat["so_trung"], hat["u"], pop["ngan_sach"], pop["h0"], tham_so, beta=beta, giu_vet=giu_vet)
    return dc, nh


def quet_beta(hat: dict, tham_so: ThamSo, betas, seed: int, n_boot: int = 500) -> list[dict]:
    # Tăng dần beta, giữ nguyên quần thể và chuỗi quay.
    rows = []
    rng = np.random.default_rng(seed + 7)
    for b in betas:
        dc, nh = chay_cap(hat, tham_so, beta=float(b))
        ss = so_sanh_cap(dc, nh, hat["so_trung"], tham_so, rng, n_boot=n_boot)
        rows.append({
            "beta": float(b),
            "chenh_ky_tb": ss["chenh_ky_tb"],
            "ci95": ss["chenh_ky_tb_ci95"],
            "tuong_doi_pct": ss["chenh_ky_tuong_doi_pct"],
            "ty_le_agent_choi_lau_hon": ss["ty_le_agent_choi_lau_hon"],
            "chenh_thua_rong_tb": ss["chenh_thua_rong_tb"],
        })
    return rows


def kiem_tra_do_ben(hat: dict, tham_so: ThamSo, beta: float, rhos, cach_gan_diem=None) -> dict:
    cach_gan_diem = list(cach_gan_diem or CACH_GAN_DIEM.keys())
    rhos = list(rhos)
    luoi_pct = np.zeros((len(cach_gan_diem), len(rhos)))
    luoi_ky = np.zeros_like(luoi_pct)
    ky_dc = np.zeros(len(rhos))
    for j, rho in enumerate(rhos):
        for i, cach in enumerate(cach_gan_diem):
            ts = replace(tham_so, rho=float(rho), cach_gan_diem=cach)
            dc, nh = chay_cap(hat, ts, beta=beta)
            d = nh["so_ky"] - dc["so_ky"]
            luoi_ky[i, j] = d.mean()
            luoi_pct[i, j] = 100 * d.mean() / dc["so_ky"].mean()
            ky_dc[j] = dc["so_ky"].mean()
    return {
        "beta": float(beta),
        "cach_gan_diem": cach_gan_diem,
        "rho": [float(r) for r in rhos],
        "chenh_ky_tb": luoi_ky.tolist(),
        "tuong_doi_pct": luoi_pct.tolist(),
        "so_ky_tb_doi_chung_theo_rho": ky_dc.tolist(),
    }


def chay_thi_nghiem(
    n_agents: int = 5000,
    tham_so: ThamSo | None = None,
    beta: float = 1.0,
    seed: int = 42,
    betas_quet=None,
    rhos_do_ben=None,
    n_boot: int = 1000,
) -> dict:
    if beta < 0:
        raise ValueError("beta phải >= 0 (beta=0 là đối chứng).")
    tham_so = tham_so or ThamSo()
    tham_so = replace(tham_so, beta=float(beta))
    hat = tao_hat_giong(n_agents, tham_so, seed)

    dc, nh = chay_cap(hat, tham_so, beta=beta, giu_vet=False)
    rng = np.random.default_rng(seed + 1)
    ss = so_sanh_cap(dc, nh, hat["so_trung"], tham_so, rng, n_boot=n_boot)

    t_max = tham_so.max_ky
    sv_dc = ty_le_con_choi(dc["so_ky"], t_max)
    sv_nh = ty_le_con_choi(nh["so_ky"], t_max)
    con_choi_moc = {
        str(k): {"doi_chung": float(sv_dc[k - 1]), "nhay": float(sv_nh[k - 1])}
        for k in MOC_KY if k <= t_max
    }

    quet = quet_beta(hat, tham_so, betas_quet, seed) if betas_quet is not None else None
    do_ben = (
        kiem_tra_do_ben(hat, tham_so, beta, rhos_do_ben) if rhos_do_ben is not None else None
    )

    cau_chuyen = dung_cau_chuyen(hat, tham_so, beta, dc, nh, seed)

    return {
        "params": {
            "n_agents": n_agents, "beta": float(beta), "seed": seed,
            **{k: v for k, v in asdict(tham_so).items() if k not in ("beta", "bang_thuong")},
            "bang_thuong": {str(k): v for k, v in tham_so.bang_thuong.items()},
        },
        "rtp_ly_thuyet": ty_le_hoan_tra_ly_thuyet(tham_so),
        "so_sanh": ss,
        "con_choi_moc": con_choi_moc,
        "survival": {"doi_chung": sv_dc, "nhay": sv_nh},
        "quet_beta": quet,
        "do_ben": do_ben,
        "cau_chuyen": cau_chuyen,
        "_hat": hat, "_dc": dc, "_nh": nh,
    }
