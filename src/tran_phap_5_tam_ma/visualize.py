import matplotlib

matplotlib.use("Agg")  # backend không cần GUI
import numpy as np
import matplotlib.pyplot as plt

MAU_DOI_CHUNG = "#4C72B0"
MAU_NHAY = "#C44E52"
MAU_PHU = "#8172B2"


def plot_con_choi(survival: dict, ax, max_hien_thi: int | None = None) -> None:
    dc, nh = survival["doi_chung"], survival["nhay"]
    if max_hien_thi is None:
        con = np.flatnonzero((dc > 0.002) | (nh > 0.002))
        max_hien_thi = int(con.max()) + 3 if con.size else len(dc)
    max_hien_thi = min(max_hien_thi, len(dc))
    x = np.arange(1, max_hien_thi + 1)
    ax.plot(x, 100 * dc[:max_hien_thi], color=MAU_DOI_CHUNG, lw=2, label="Đối chứng (β=0)")
    ax.plot(x, 100 * nh[:max_hien_thi], color=MAU_NHAY, lw=2, label="Nhạy suýt trúng")
    ax.fill_between(x, 100 * dc[:max_hien_thi], 100 * nh[:max_hien_thi], color=MAU_NHAY, alpha=0.2)
    ax.set_xlabel("Kỳ quay")
    ax.set_ylabel("% agent còn mua vé")
    ax.set_title("Tỷ lệ còn chơi theo thời gian", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_chi_nhan(so_sanh: dict, ax) -> None:
    nhan_x = ["Đối chứng", "Nhạy suýt trúng"]
    chi = np.array([so_sanh["chi_tb_doi_chung"], so_sanh["chi_tb_nhay"]]) / 1000
    nhan = np.array([so_sanh["nhan_tb_doi_chung"], so_sanh["nhan_tb_nhay"]]) / 1000
    ky = [so_sanh["so_ky_tb_doi_chung"], so_sanh["so_ky_tb_nhay"]]
    x = np.arange(2)
    w = 0.36
    b1 = ax.bar(x - w / 2, chi, w, color=MAU_PHU, alpha=0.85, label="Tiền chi")
    b2 = ax.bar(x + w / 2, nhan, w, color="#55A868", alpha=0.85, label="Tiền nhận")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.1f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\n({k:.2f} kỳ TB)" for n, k in zip(nhan_x, ky)])
    ax.set_ylabel("Nghìn đồng / agent")
    lo, hi = so_sanh["chenh_ky_tb_ci95"]
    ax.set_title(
        f"Chi - nhận trung bình mỗi agent\nchênh {so_sanh['chenh_ky_tb']:+.2f} kỳ (CI95 {lo:+.2f}..{hi:+.2f})",
        fontsize=10,
    )
    ax.legend(fontsize=8)


def plot_cau_chuyen(cau_chuyen: dict | None, ax) -> None:
    if cau_chuyen is None:
        ax.text(0.5, 0.5, "Không có agent nào bị kéo dài\n(β = 0 hoặc quá nhỏ)", ha="center", va="center")
        ax.set_axis_off()
        return
    cac_ky = cau_chuyen["cac_ky"]
    ky = [c["ky"] for c in cac_ky]
    h_dc = [np.nan if c["h_doi_chung"] is None else c["h_doi_chung"] for c in cac_ky]
    h_nh = [c["h_nhay"] for c in cac_ky]
    ax.plot(ky, h_dc, color=MAU_DOI_CHUNG, lw=2, marker="o", ms=3, label="Đối chứng")
    ax.plot(ky, h_nh, color=MAU_NHAY, lw=2, marker="o", ms=3, label="Nhạy suýt trúng")
    gap = [c for c in cac_ky if c["diem_suyt_trung"] > 0]
    if gap:
        ax.scatter([c["ky"] for c in gap], [c["h_nhay"] for c in gap], marker="*", s=180,
                   color="gold", edgecolor="black", zorder=5, label="Suýt trúng (trùng 3-5 số)")
    ax.axvline(cau_chuyen["ky_dung_doi_chung"], color=MAU_DOI_CHUNG, ls=":", lw=1.2)
    ax.axvline(cau_chuyen["ky_dung_nhay"], color=MAU_NHAY, ls=":", lw=1.2)
    ax.set_xlabel("Kỳ quay")
    ax.set_ylabel("Hứng thú H")
    ax.set_title(f"Câu chuyện agent #{cau_chuyen['agent']} (đại diện, không phải điển hình nhất)", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def build_main_figure(ket_qua: dict):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    plot_con_choi(ket_qua["survival"], axes[0])
    plot_chi_nhan(ket_qua["so_sanh"], axes[1])
    plot_cau_chuyen(ket_qua["cau_chuyen"], axes[2])
    fig.suptitle(
        f"Trận Pháp 5: Tâm Ma Suýt Trúng (β={ket_qua['params']['beta']:g}, "
        f"{ket_qua['params']['n_agents']:,} agent, mỗi agent chạy hai bản ghép cặp)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    return fig


def plot_quet_beta(quet: list[dict], ax) -> None:
    b = np.array([r["beta"] for r in quet])
    d = np.array([r["chenh_ky_tb"] for r in quet])
    lo = np.array([r["ci95"][0] for r in quet])
    hi = np.array([r["ci95"][1] for r in quet])
    ax.plot(b, d, color=MAU_NHAY, lw=2, marker="o")
    ax.fill_between(b, lo, hi, color=MAU_NHAY, alpha=0.2, label="CI95 (bootstrap theo agent)")
    ax.set_xlabel("β (độ nhạy với suýt trúng)")
    ax.set_ylabel("Số kỳ chơi thêm trung bình", color=MAU_NHAY)
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(b, [100 * r["ty_le_agent_choi_lau_hon"] for r in quet], color=MAU_PHU, ls="--", marker="s", ms=4)
    ax2.set_ylabel("% agent chơi lâu hơn bản đối chứng", color=MAU_PHU)
    ax.set_title("Tăng dần β: kéo dài bao nhiêu?", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")


def plot_do_ben(do_ben: dict, ax) -> None:
    luoi = np.array(do_ben["tuong_doi_pct"])
    im = ax.imshow(luoi, cmap="OrRd", aspect="auto")
    ax.set_xticks(range(len(do_ben["rho"])))
    ax.set_xticklabels([f"ρ={r:g}\n({k:.1f} kỳ TB)" for r, k in
                        zip(do_ben["rho"], do_ben["so_ky_tb_doi_chung_theo_rho"])], fontsize=8)
    ax.set_yticks(range(len(do_ben["cach_gan_diem"])))
    ax.set_yticklabels(do_ben["cach_gan_diem"], fontsize=8)
    for i in range(luoi.shape[0]):
        for j in range(luoi.shape[1]):
            ax.text(j, i, f"{luoi[i, j]:.1f}%", ha="center", va="center", fontsize=9,
                    color="white" if luoi[i, j] > luoi.max() / 2 else "black")
    ax.set_title(f"Độ bền: % kéo dài tương đối (β={do_ben['beta']:g})\n"
                 "theo cách gán điểm S và ρ", fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def build_sweep_figure(quet: list[dict] | None, do_ben: dict | None):
    n = int(quet is not None) + int(do_ben is not None)
    if n == 0:
        return None
    fig, axes = plt.subplots(1, n, figsize=(7.5 * n, 5.2))
    axes = np.atleast_1d(axes)
    i = 0
    if quet is not None:
        plot_quet_beta(quet, axes[i]); i += 1
    if do_ben is not None:
        plot_do_ben(do_ben, axes[i])
    fig.suptitle("Trận Pháp 5: nhạy đến đâu với giả định về suýt trúng?", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig


def luu_hinh(fig, save_path) -> None:
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
