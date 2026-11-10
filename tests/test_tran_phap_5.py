import json
from dataclasses import replace
from math import comb

import numpy as np

from src.tran_phap_5_near_miss.config import ThamSo, CACH_GAN_DIEM
from src.tran_phap_5_near_miss.draws import (
    dung_ky_quay_co_k_trung, pmf_so_trung, quay_so_that, sample_so_trung, so_trung,
)
from src.tran_phap_5_near_miss.dynamics import (
    CHAM_GIOI_HAN, HET_NGAN_SACH, bang_diem_suyt_trung, bang_thuong_theo_so_trung,
    chay_mot_ban, chuan_hoa_tien,
)
from src.tran_phap_5_near_miss.simulate import chay_cap, chay_thi_nghiem, tao_hat_giong
from src.tran_phap_5_near_miss.stats import summarize, dem_suyt_trung, so_sanh_cap, ty_le_con_choi
from src.tran_phap_5_near_miss.story import dung_cau_chuyen


def test_pmf_so_trung_khop_cong_thuc_to_hop():
    pmf = pmf_so_trung()
    assert abs(pmf.sum() - 1.0) < 1e-12
    assert abs(pmf[6] - 1 / comb(55, 6)) < 1e-18
    assert abs(pmf[5] - 6 * 49 / comb(55, 6)) < 1e-15


def test_lay_so_trung_truc_tiep_khop_quay_so_that():
    # Quay 6/55 thật rồi đếm trùng với một vé cố định phải cùng phân phối với sampler siêu bội.
    rng = np.random.default_rng(0)
    ve = np.array([3, 11, 20, 29, 40, 52])
    n = 100_000
    that = np.bincount([so_trung(ve, quay_so_that(rng)) for _ in range(n)], minlength=7) / n
    nhanh = np.bincount(sample_so_trung(np.random.default_rng(1), n), minlength=7) / n
    pmf = pmf_so_trung()
    assert np.all(np.abs(that[:4] - pmf[:4]) < 0.006)
    assert np.all(np.abs(nhanh[:4] - pmf[:4]) < 0.006)


def test_dung_ky_quay_co_dung_k_so_trung():
    rng = np.random.default_rng(3)
    ve = np.array([5, 9, 17, 30, 44, 55])
    for k in range(0, 7):
        quay = dung_ky_quay_co_k_trung(rng, ve, k)
        assert len(set(quay.tolist())) == 6
        assert quay.min() >= 1 and quay.max() <= 55
        assert so_trung(ve, quay) == k


def test_cac_ky_quay_doc_lap_suyt_trung_khong_bao_truoc_ky_sau():
    # Điều agent "cảm thấy" không có thật: P(suýt trúng kỳ sau | suýt trúng kỳ này) = P(suýt trúng).
    m = sample_so_trung(np.random.default_rng(5), (200, 5000))
    gap = (m >= 3) & (m <= 5)
    p = gap.mean()
    p_cond = gap[:, 1:][gap[:, :-1]].mean()
    assert abs(p_cond - p) < 0.004


def test_bang_diem_khong_tinh_jackpot_va_trung_it_la_suyt_trung():
    for ten in CACH_GAN_DIEM:
        s = bang_diem_suyt_trung(ten)
        assert s[6] == 0.0 and s[0] == s[1] == s[2] == 0.0
        assert s[5] == 1.0  # sát jackpot nhất luôn là mức cao nhất
        assert s[3] <= s[4] <= s[5]


def test_chuan_hoa_tien_thua_am_thang_duong_va_bi_chan():
    r = chuan_hoa_tien(np.array([-1.0, 4.0, 1e9]), r_cap=5.0)
    assert r[0] < 0 < r[1]
    assert r[2] == 5.0


def test_beta_bang_0_ban_nhay_trung_hoan_toan_doi_chung():
    ts = ThamSo()
    hat = tao_hat_giong(2000, ts, seed=7)
    dc, nh = chay_cap(hat, ts, beta=0.0)
    for k in ("so_ky", "chi", "nhan", "ly_do"):
        assert np.array_equal(dc[k], nh[k])


def test_ghep_cap_ban_nhay_khong_bao_gio_choi_it_hon():
    ts = ThamSo()
    hat = tao_hat_giong(3000, ts, seed=8)
    for beta in (0.25, 1.0, 3.0):
        dc, nh = chay_cap(hat, ts, beta=beta)
        d = nh["so_ky"] - dc["so_ky"]
        assert (d >= 0).all()
        assert d.sum() > 0
        # tiền chi chỉ chênh vì số kỳ chơi thêm
        assert np.allclose(nh["chi"] - dc["chi"], ts.gia_ve * d)


def test_beta_lon_hon_thi_moi_agent_choi_khong_it_hon():
    ts = ThamSo()
    hat = tao_hat_giong(3000, ts, seed=9)
    pop = hat["pop"]
    ky = [chay_mot_ban(hat["so_trung"], hat["u"], pop["ngan_sach"], pop["h0"], ts, b)["so_ky"]
          for b in (0.0, 0.5, 1.0, 2.0)]
    for a, b in zip(ky[:-1], ky[1:]):
        assert (b >= a).all()


def test_hai_ban_nhan_thuong_nhu_nhau_theo_so_trung():
    ts = ThamSo()
    hat = tao_hat_giong(2000, ts, seed=10)
    thuong = bang_thuong_theo_so_trung(ts)
    for beta in (0.0, 2.0):
        r = chay_mot_ban(hat["so_trung"], hat["u"], hat["pop"]["ngan_sach"], hat["pop"]["h0"], ts, beta)
        t_max = hat["so_trung"].shape[1]
        da_choi = np.arange(t_max)[None, :] < r["so_ky"][:, None]
        mong_doi = (thuong[hat["so_trung"]] * da_choi).sum(axis=1)
        assert np.allclose(r["nhan"], mong_doi)


def test_bat_bien_ngan_sach_va_ly_do_dung():
    ts = ThamSo(ngan_sach_median_ve=5, max_ky=40)
    hat = tao_hat_giong(3000, ts, seed=11)
    pop = hat["pop"]
    r = chay_mot_ban(hat["so_trung"], hat["u"], pop["ngan_sach"], pop["h0"], ts, 1.0)
    assert np.allclose(r["chi"], ts.gia_ve * r["so_ky"])
    assert (r["chi"] <= pop["ngan_sach"] + r["nhan"] + 1e-9).all()  # không bao giờ chi âm vốn
    assert (r["so_ky"] <= ts.max_ky).all()
    assert (r["so_ky"][r["ly_do"] == CHAM_GIOI_HAN] == ts.max_ky).all()
    assert (r["ly_do"] == HET_NGAN_SACH).any()


def test_agent_khong_du_tien_mua_ve_dau_thi_khong_choi():
    ts = ThamSo(max_ky=10)
    so_trung_ = np.zeros((1, 10), dtype=np.int8)
    r = chay_mot_ban(so_trung_, np.zeros((1, 10)), np.array([ts.gia_ve - 1]), np.array([5.0]), ts, 1.0)
    assert r["so_ky"][0] == 0 and r["ly_do"][0] == HET_NGAN_SACH


def test_tai_lap_theo_seed():
    a = chay_thi_nghiem(1500, ThamSo(), beta=1.0, seed=3, n_boot=100)
    b = chay_thi_nghiem(1500, ThamSo(), beta=1.0, seed=3, n_boot=100)
    c = chay_thi_nghiem(1500, ThamSo(), beta=1.0, seed=4, n_boot=100)
    assert a["so_sanh"]["chenh_ky_tb"] == b["so_sanh"]["chenh_ky_tb"]
    assert a["so_sanh"]["chenh_ky_tb"] != c["so_sanh"]["chenh_ky_tb"]


def test_beta_am_bi_tu_choi():
    try:
        chay_thi_nghiem(100, ThamSo(), beta=-0.1)
    except ValueError:
        return
    raise AssertionError("beta < 0 phải báo lỗi")


def test_chi_agent_tung_gap_suyt_trung_moi_bi_keo_dai():
    kq = chay_thi_nghiem(4000, ThamSo(), beta=1.0, seed=12, n_boot=100)
    ss = kq["so_sanh"]
    assert ss["chenh_ky_tb_nhom_khong_gap"] == 0.0
    assert ss["chenh_ky_tb_nhom_tung_gap"] > 0.0
    assert ss["chenh_ky_tb_ci95"][0] > 0.0  # ở beta=1 với 4000 agent, hiệu ứng vượt nhiễu bootstrap


def test_cau_chuyen_chi_co_khi_beta_duong_va_dung_agent_bi_keo_dai():
    ts = ThamSo()
    hat = tao_hat_giong(3000, ts, seed=13)
    dc0, nh0 = chay_cap(hat, ts, beta=0.0)
    assert dung_cau_chuyen(hat, ts, 0.0, dc0, nh0, 13) is None
    dc, nh = chay_cap(hat, ts, beta=2.0)
    ct = dung_cau_chuyen(hat, ts, 2.0, dc, nh, 13)
    i = ct["agent"]
    assert nh["so_ky"][i] > dc["so_ky"][i]
    assert ct["ky_dung_nhay"] == nh["so_ky"][i] and ct["ky_dung_doi_chung"] == dc["so_ky"][i]
    # Hai bản chỉ khác nhau qua beta*S, nên agent bị kéo dài buộc phải có ít nhất một kỳ suýt trúng
    assert any(c["diem_suyt_trung"] > 0 for c in ct["cac_ky"])


def test_summarize_la_json_thuan():
    kq = chay_thi_nghiem(1500, ThamSo(), beta=1.0, seed=14, betas_quet=[0.0, 1.0], rhos_do_ben=[0.97, 0.985], n_boot=100)
    json.dumps(summarize(kq))  # không dùng default=str: phải là kiểu Python thuần
    assert len(kq["do_ben"]["tuong_doi_pct"]) == len(CACH_GAN_DIEM)
    assert kq["quet_beta"][0]["chenh_ky_tb"] == 0.0


def test_dem_suyt_trung_theo_bang_diem_va_chi_dem_ky_da_choi():
    m = np.array([[3, 4, 5, 6], [3, 5, 4, 6], [6, 0, 1, 2]], dtype=np.int8)
    so_ky = np.array([2, 1, 4])
    assert np.array_equal(dem_suyt_trung(m, so_ky), [2, 1, 0])
    for ten in CACH_GAN_DIEM:
        bang = bang_diem_suyt_trung(ten)
        mong_doi = [sum(bang[int(k)] > 0 for k in row[:int(n)])
                    for row, n in zip(m, so_ky)]
        assert np.array_equal(dem_suyt_trung(m, so_ky, bang_diem=bang), mong_doi)
    assert np.array_equal(
        dem_suyt_trung(m, so_ky, bang_diem=bang_diem_suyt_trung('chi_4_5_so')), [1, 0, 0]
    )


def test_so_sanh_phan_nhom_theo_tin_hieu_cau_hinh():
    # Agent 0 chỉ trùng 3: phải thuộc nhóm S=0 khi dùng chi_4_5_so.
    # Agent 1 trùng 4: thuộc nhóm có tín hiệu. Agent 2 chưa từng gặp.
    ts = ThamSo(max_ky=2, cach_gan_diem='chi_4_5_so')
    m = np.array([[3, 0], [4, 0], [0, 0]], dtype=np.int8)
    u = np.full((3, 2), 0.9999)
    tien = np.full(3, 1_000_000)
    h0 = np.full(3, 3.0)
    dc = chay_mot_ban(m, u, tien, h0, ts, beta=0.0)
    nh = chay_mot_ban(m, u, tien, h0, ts, beta=1.0)
    ss = so_sanh_cap(dc, nh, m, ts, np.random.default_rng(0), n_boot=20)
    assert ss['ty_le_agent_tung_gap_suyt_trung'] == 1 / 3
    assert ss['chenh_ky_tb_nhom_khong_gap'] == 0.0


def test_duong_con_choi_dem_o_ky_khong_phai_sau_ky():
    # Một agent dừng sau kỳ 1, một sau kỳ 2, một được theo dõi hết kỳ 3.
    curve = ty_le_con_choi(np.array([1, 2, 3]), t_max=3)
    assert np.allclose(curve, [1.0, 2 / 3, 1 / 3])
    assert curve[1] == 2 / 3  # mua kỳ 2
    assert curve[2] == 1 / 3  # tiếp tục sau kỳ 2, sang kỳ 3
