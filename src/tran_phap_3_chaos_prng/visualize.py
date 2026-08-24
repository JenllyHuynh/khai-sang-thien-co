import matplotlib

matplotlib.use("Agg")  # backend không cần GUI
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - cần import để kích hoạt projection="3d"

MAU_LORENZ = "#4C72B0"
MAU_WEAK = "#C44E52"
MAU_STRONG = "#55A868"

# Góc nhìn 3D để lộ rõ "siêu phẳng" của RANDU (Marsaglia 1968): với quan hệ x_{n+2} = 6*x_{n+1} - 9*x_n (mod m),
# pháp tuyến của họ mặt phẳng song song là (9, -6, 1)
# Để THẤY các mặt phẳng dạng lát mỏng xếp chồng (chứ không nhìn thẳng vào mặt phẳng)
# camera phải nhìn DỌC THEO một vector NẰM TRONG mặt phẳng đó (vuông góc với pháp tuyến), ví dụ (6, 9, 0)
_LATTICE_NORMAL = np.array([9.0, -6.0, 1.0])
_VIEW_DIRECTION = np.array([6.0, 9.0, 0.0])  # vuông góc _LATTICE_NORMAL
assert abs(np.dot(_LATTICE_NORMAL, _VIEW_DIRECTION)) < 1e-9
_VIEW_DIRECTION = _VIEW_DIRECTION / np.linalg.norm(_VIEW_DIRECTION)
LATTICE_VIEW_AZIM = float(np.degrees(np.arctan2(_VIEW_DIRECTION[1], _VIEW_DIRECTION[0])))
LATTICE_VIEW_ELEV = 8.0  # nghiêng nhẹ để có chiều sâu 3D, vẫn giữ rõ cấu trúc

def plot_chaos_summary(trajectory, times, distance, lyapunov: dict, save_path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].plot(trajectory[:, 0], trajectory[:, 2], linewidth=0.4, color=MAU_LORENZ)
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("z")
    axes[0].set_title("Trận Đồ Hỗn Mang (Lorenz Attractor)")

    axes[1].semilogy(times, distance, color=MAU_LORENZ, linewidth=1)
    axes[1].set_xlabel("Thời gian")
    axes[1].set_ylabel("Khoảng cách giữa 2 quỹ đạo (thang log)")
    title = "Hiệu Ứng Cánh Bướm"
    if lyapunov.get("doubling_time"):
        title += f"\n(sai số 1e-8 nhân đôi mỗi ~{lyapunov['doubling_time']:.2f} đơn vị thời gian)"
    axes[1].set_title(title, fontsize=10)

    fig.suptitle(
        "Trận Pháp 3 (Phần 1/2): Hỗn Mang Có Quy Luật - Lorenz Attractor",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

def plot_prng_structure(weak_samples, strong_samples, save_path) -> None:
    fig = plt.figure(figsize=(13, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    x1, y1, z1 = weak_samples[:-2], weak_samples[1:-1], weak_samples[2:]
    ax1.scatter(x1, y1, z1, s=2, alpha=0.5, color=MAU_WEAK)
    ax1.set_title("LCG Yếu (kiểu RANDU)\n'Yêu quái tầm thường' - lộ rõ 15 siêu phẳng song song", fontsize=9)
    ax1.set_xlabel("$x_i$", fontsize=8)
    ax1.set_ylabel("$x_{i+1}$", fontsize=8)
    ax1.set_zlabel("$x_{i+2}$", fontsize=8)
    ax1.view_init(elev=LATTICE_VIEW_ELEV, azim=LATTICE_VIEW_AZIM)

    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    x2, y2, z2 = strong_samples[:-2], strong_samples[1:-1], strong_samples[2:]
    ax2.scatter(x2, y2, z2, s=2, alpha=0.5, color=MAU_STRONG)
    ax2.set_title("CSPRNG Mạnh (PCG64 - numpy)\n'Thần tiên chính thống' - cùng góc nhìn, không thấy cấu trúc", fontsize=9)
    ax2.set_xlabel("$x_i$", fontsize=8)
    ax2.set_ylabel("$x_{i+1}$", fontsize=8)
    ax2.set_zlabel("$x_{i+2}$", fontsize=8)
    ax2.view_init(elev=LATTICE_VIEW_ELEV, azim=LATTICE_VIEW_AZIM)

    fig.suptitle(
        "Trận Pháp 3 (Phần 2/2): Bắt Mạch Yêu Quái - Cấu Trúc PRNG Yếu vs Mạnh",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
