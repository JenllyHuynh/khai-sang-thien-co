import matplotlib

matplotlib.use("Agg")  # backend không cần GUI, an toàn cho mọi máy
import matplotlib.pyplot as plt

MAU_VO_VI = "#4C72B0"
MAU_TINH_CAM = "#DD8452"

def build_summary_figure(stats_vo_vi: dict, stats_tinh_cam: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["Vô Vi\n(random thuần)", "Tình Cảm\n(số đẹp/ngày sinh)"]
    values = [
        stats_vo_vi["pct_players_sharing_ticket"],
        stats_tinh_cam["pct_players_sharing_ticket"],
    ]
    bars = axes[0].bar(labels, values, color=[MAU_VO_VI, MAU_TINH_CAM])
    axes[0].set_ylabel("% người chơi trùng vé với người khác")
    axes[0].set_title("Nguy cơ phải chia sẻ giải thưởng")
    for bar, v in zip(bars, values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2, v, f"{v:.4f}%",
            ha="center", va="bottom", fontsize=9,
        )

    labels2 = ["Vô Vi", "Tình Cảm"]
    values2 = [stats_vo_vi["max_cluster_size"], stats_tinh_cam["max_cluster_size"]]
    bars2 = axes[1].bar(labels2, values2, color=[MAU_VO_VI, MAU_TINH_CAM])
    axes[1].set_ylabel("Số người trùng vé đông nhất (1 tổ hợp)")
    axes[1].set_title("Cụm vé phổ biến nhất")
    for bar, v in zip(bars2, values2):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, v, f"{v:,}",
            ha="center", va="bottom", fontsize=9,
        )

    fig.suptitle("Trận Pháp 1: Vạn Kiếp Quy Tông - Vô Vi vs Tình Cảm", fontsize=13, fontweight="bold")
    plt.tight_layout()
    return fig

def build_repeated_runs_figure(df, agg: dict):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(df["run_index"], df["vo_vi_pct_sharing"], "o-", color=MAU_VO_VI,
                 label="Vô Vi", markersize=3, linewidth=1)
    axes[0].plot(df["run_index"], df["tinh_cam_pct_sharing"], "o-", color=MAU_TINH_CAM,
                 label="Tình Cảm", markersize=3, linewidth=1)
    axes[0].set_xlabel("Lần chạy thứ #")
    axes[0].set_ylabel("% người chơi trùng vé")
    axes[0].set_title(f"Ổn định qua {len(df)} lần chạy độc lập")
    axes[0].legend()

    axes[1].boxplot(
        [df["vo_vi_pct_sharing"], df["tinh_cam_pct_sharing"]],
        tick_labels=["Vô Vi", "Tình Cảm"],
        patch_artist=True,
        boxprops=dict(facecolor=MAU_VO_VI, alpha=0.5),
    )
    axes[1].set_ylabel("% người chơi trùng vé")
    axes[1].set_title("Phân phối qua tất cả các lần chạy")

    total = agg["total_players_simulated_per_group"]
    fig.suptitle(
        f"Trận Pháp 1 (Repeated Trials): {agg['n_runs']} lần chạy x "
        f"~{total // agg['n_runs']:,} vé/phe/lần = {total:,} vé/phe tổng cộng",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    return fig

def plot_summary(stats_vo_vi: dict, stats_tinh_cam: dict, save_path) -> None:
    fig = build_summary_figure(stats_vo_vi, stats_tinh_cam)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_repeated_runs(df, agg: dict, save_path) -> None:
    fig = build_repeated_runs_figure(df, agg)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
