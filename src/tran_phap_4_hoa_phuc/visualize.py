import matplotlib

matplotlib.use("Agg")  # backend không cần GUI
import numpy as np
import matplotlib.pyplot as plt

MAU_TRUNG = "#DD8452"
MAU_KHONG_TRUNG = "#4C72B0"
MAU_NULL = "#8172B2"
MAU_HIGHLIGHT = "#C44E52"

# Vẽ ma trận 2x2 dạng heatmap có chú thích số lượng + phần trăm
def plot_contingency_heatmap(table: np.ndarray, ax) -> None:
    total = table.sum()
    labels_row = ["Trúng Jackpot (A)", "Không Trúng"]
    labels_col = ["Gặp Họa (B)", "Không Họa"]

    im = ax.imshow(table, cmap="OrRd", aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(labels_col)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(labels_row)
    ax.set_title("Bàn Cờ Nhân Quả (số người)")

    for i in range(2):
        for j in range(2):
            count = table[i, j]
            pct = 100 * count / total
            ax.text(
                j, i, f"{count:,}\n({pct:.3f}%)",
                ha="center", va="center", fontsize=9,
                color="white" if count > table.max() / 2 else "black",
            )
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

def plot_conditional_comparison(independence_result: dict, ax) -> None:
    p_a1 = independence_result["p_hoa_given_trung"]
    p_a0 = independence_result["p_hoa_given_khong_trung"]
    n_a1 = independence_result["n_a1"]
    n_a0 = independence_result["n_a0"]

    se_a1 = np.sqrt(p_a1 * (1 - p_a1) / n_a1) if n_a1 > 0 else 0
    se_a0 = np.sqrt(p_a0 * (1 - p_a0) / n_a0) if n_a0 > 0 else 0
    ci_a1 = 1.96 * se_a1 * 100
    ci_a0 = 1.96 * se_a0 * 100

    labels = ["Người TRÚNG số", "Người KHÔNG trúng"]
    values = [p_a1 * 100, p_a0 * 100]
    errors = [ci_a1, ci_a0]
    colors = [MAU_TRUNG, MAU_KHONG_TRUNG]

    bars = ax.bar(labels, values, yerr=errors, capsize=6, color=colors, alpha=0.85)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}%", ha="center", va="bottom", fontsize=9)

    p_val = independence_result["p_value"]
    y_nghia = "CÓ ý nghĩa thống kê" if independence_result["co_y_nghia_thong_ke"] else "KHÔNG có ý nghĩa thống kê"
    ax.set_ylabel("Tỷ lệ gặp họa (%)")
    ax.set_title(f"P(Họa | Trúng) vs P(Họa | Không Trúng)\n(Chi² p-value={p_val:.3f} - {y_nghia})", fontsize=10)

def plot_null_distribution(null_diffs: np.ndarray, actual_diff: float, ax) -> None:
    ax.hist(null_diffs, bins=30, color=MAU_NULL, alpha=0.75, edgecolor="white")
    ax.axvline(0, color="black", linestyle="-", linewidth=1, label="Không chênh lệch (độc lập hoàn hảo)")
    ax.axvline(
        actual_diff, color=MAU_HIGHLIGHT, linestyle="--", linewidth=2,
        label=f"Kết quả lần chạy chính ({actual_diff:+.3f} điểm %)",
    )
    ax.set_xlabel("Chênh lệch P(Họa|Trúng) − P(Họa|Không Trúng) (điểm %)")
    ax.set_ylabel(f"Số lần chạy (trong {len(null_diffs)} lần độc lập)")
    ax.set_title("Phân Phối 'Nhiễu Do May Rủi' Qua Nhiều Lần Chạy Độc Lập", fontsize=10)
    ax.legend(fontsize=8)


def plot_full_summary(table, independence_result, null_diffs, save_path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    plot_contingency_heatmap(table, axes[0])
    plot_conditional_comparison(independence_result, axes[1])
    plot_null_distribution(null_diffs, independence_result["chenh_lech_diem_phan_tram"], axes[2])

    fig.suptitle(
        "Trận Pháp 4: Bàn Cờ Nhân Quả - Họa Phúc Song Hành (Quantifying Luck)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
