
import matplotlib

matplotlib.use("Agg")  # backend không cần GUI
import numpy as np
import matplotlib.pyplot as plt

MAU_TRAIN = "#4C72B0"
MAU_TEST = "#C44E52"

def build_overfitting_figure(summary: dict):
    models = list(summary["models"].keys())
    train_matches = [summary["models"][m]["train"]["avg_matches"] for m in models]
    test_matches = [summary["models"][m]["test"]["avg_matches"] for m in models]
    train_conf = [summary["models"][m]["train"]["avg_confidence"] * 100 for m in models]
    test_conf = [summary["models"][m]["test"]["avg_confidence"] * 100 for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(models))
    width = 0.35

    # Trái: số trúng trung bình (train vs test)
    bars_a = axes[0].bar(x - width / 2, train_matches, width, label="Trên dữ liệu ĐÃ HỌC (train)", color=MAU_TRAIN)
    bars_b = axes[0].bar(x + width / 2, test_matches, width, label="Trên TƯƠNG LAI chưa thấy (test)", color=MAU_TEST)
    axes[0].axhline(
        summary["theoretical_random_matches"], color="gray", linestyle="--", linewidth=1.5,
        label=f"Đoán mò lý thuyết ({summary['theoretical_random_matches']:.2f})",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, fontsize=9)
    axes[0].set_ylabel("Số trúng trung bình / 6 số dự đoán")
    axes[0].set_title("Số Trúng: Học Vẹt vs Thực Tế")
    axes[0].legend(fontsize=8)
    for bars in (bars_a, bars_b):
        for bar in bars:
            h = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width() / 2, h, f"{h:.2f}", ha="center", va="bottom", fontsize=8)

    # Phải: độ tự tin trung bình (train vs test), kèm marker "sự thật trần trụi"
    bars_c = axes[1].bar(x - width / 2, train_conf, width, label="Độ tự tin AI tự nhận (train)", color=MAU_TRAIN)
    bars_d = axes[1].bar(x + width / 2, test_conf, width, label="Độ tự tin AI tự nhận (test)", color=MAU_TEST)

    # "Sự thật trần trụi": tỷ lệ THỰC SỰ đúng trong 6 số AI chọn (so_khop / PICK).
    # Đặt chồng lên cột độ tự tin để lộ rõ khoảng cách giữa "AI nghĩ nó đúng"
    # và "AI thực sự đúng" - đây chính là hình ảnh của sự "ảo tưởng".
    real_accuracy_train = [m / 6 * 100 for m in train_matches]
    real_accuracy_test = [m / 6 * 100 for m in test_matches]
    axes[1].scatter(x - width / 2, real_accuracy_train, marker="*", s=180, color="gold",
                     edgecolors="black", linewidths=0.8, zorder=5, label="Sự thật trần trụi (số trúng/6)")
    axes[1].scatter(x + width / 2, real_accuracy_test, marker="*", s=180, color="gold",
                     edgecolors="black", linewidths=0.8, zorder=5)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, fontsize=9)
    axes[1].set_ylabel("Phần trăm (%)")
    axes[1].set_title("Ảo Tưởng Tự Tin vs Sự Thật Trần Trụi")
    axes[1].legend(fontsize=8, loc="upper right")
    for bars in (bars_c, bars_d):
        for bar in bars:
            h = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width() / 2, h, f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Trận Pháp 2: Lão Tặc AI - Kẻ Học Vẹt", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig

def plot_overfitting_summary(summary: dict, save_path) -> None:
    fig = build_overfitting_figure(summary)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
