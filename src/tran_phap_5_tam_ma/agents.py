import numpy as np

from .config import N_NUMBERS, PICK, ThamSo


def tao_quan_the(n_agents: int, tham_so: ThamSo, rng: np.random.Generator) -> dict:
    ve_so = tham_so.ngan_sach_median_ve * np.exp(tham_so.ngan_sach_sigma * rng.standard_normal(n_agents))
    so_ve = np.maximum(1, np.round(ve_so)).astype(np.int64)
    ngan_sach = so_ve * tham_so.gia_ve

    bo_so = np.sort(
        np.argsort(rng.random((n_agents, N_NUMBERS)), axis=1)[:, :PICK] + 1, axis=1
    )

    h0 = tham_so.h0_mean + tham_so.h0_std * rng.standard_normal(n_agents)

    return {"ngan_sach": ngan_sach, "bo_so": bo_so, "h0": h0}
