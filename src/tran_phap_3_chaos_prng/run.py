import argparse
import json
import time
from pathlib import Path
from  datetime import datetime

from .simulate import run_simulation
from .visualize import plot_chaos_summary, plot_prng_structure

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trận Pháp 3: Phá Giải Hỗn Mang (Chaos vs. PRNG)"
    )
    parser.add_argument("--n-steps-attractor", type=int, default=6000,
                         help="Số bước tích phân RK4 để vẽ trận đồ Lorenz.")
    parser.add_argument("--dt", type=float, default=0.01, help="Bước thời gian tích phân.")
    parser.add_argument("--n-steps-butterfly", type=int, default=3000,
                         help="Số bước mô phỏng hiệu ứng cánh bướm.")
    parser.add_argument("--n-prng-samples", type=int, default=5000,
                         help="Số mẫu sinh từ mỗi loại PRNG để so sánh cấu trúc.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/tran_phap_3")

    parser.add_argument(
        "--no-timestamp", action="store_true",
        help="Tắt timestamp"
    )
    return parser

def main() -> None:
    args = build_parser().parse_args()
    
    if args.no_timestamp:
        out_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"{args.output_dir}_{timestamp}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Lưu kết quả vào -> {out_dir}\n")

    print("  Khởi động Trận Pháp 3: Phá Giải Hỗn Mang...\n")

    t0 = time.time()
    result = run_simulation(
        n_steps_attractor=args.n_steps_attractor,
        dt=args.dt,
        n_steps_butterfly=args.n_steps_butterfly,
        n_prng_samples=args.n_prng_samples,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    print(f"  Mô phỏng xong trong {elapsed:.1f} giây.\n")

    print(" Chân Lý Hé Lộ:\n")

    lyap = result["lyapunov"]
    if lyap.get("doubling_time"):
        print(
            f"    Hiệu ứng cánh bướm: sai số khởi tạo 1e-8 nhân đôi mỗi "
            f"~{lyap['doubling_time']:.2f} đơn vị thời gian"
        )
        print(f"     (số mũ Lyapunov ước lượng ~ {lyap['slope']:.4f}; "
              f"giá trị lý thuyết tham khảo cho hệ Lorenz cổ điển ~ 0.9056)")
    print()

    crack_weak = result["crack_weak_result"]
    crack_strong = result["naive_crack_on_strong"]
    print(f"    Bắt mạch LCG yếu (RANDU): chỉ 1 số quan sát, dự đoán {crack_weak['n_predicted']} số tiếp theo")
    print(
        f"     -> sai số tối đa: {crack_weak['max_abs_error']:.2e} "
        f"({'CRACK THÀNH CÔNG' if crack_weak['cracked'] else 'thất bại'})"
    )
    print(f"    Thử ĐÚNG kỹ thuật đó lên CSPRNG mạnh (PCG64):")
    print(
        f"     -> sai số tối đa: {crack_strong['max_abs_error']:.2e} "
        f"({'crack thành công (?!)' if crack_strong['cracked'] else 'THẤT BẠI HOÀN TOÀN  (đúng như kỳ vọng)'})"
    )

    plot1_path = out_dir / "hon_mang_lorenz.png"
    plot_chaos_summary(result["trajectory"], result["butterfly_times"], result["butterfly_distance"], lyap, plot1_path)
    print(f"\n  Đã lưu biểu đồ Hỗn Mang -> {plot1_path}")

    plot2_path = out_dir / "cau_truc_prng.png"
    plot_prng_structure(result["weak_lcg_samples"], result["strong_csprng_samples"], plot2_path)
    print(f"  Đã lưu biểu đồ Cấu Trúc PRNG -> {plot2_path}")

    summary = {
        "lyapunov": lyap,
        "crack_weak_lcg": crack_weak,
        "naive_crack_on_strong_csprng": crack_strong,
    }
    summary_path = out_dir / "tong_ket.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f" Đã lưu tổng kết JSON -> {summary_path}")

    print(
        "\n   Trận Pháp 3 hoàn thành! Ngộ ra: Hỗn mang (chaos) VẪN có quy luật/cấu trúc"
        "\n   tất định - chỉ là cực nhạy với điều kiện ban đầu; còn PRNG yếu thì lộ cấu"
        "\n   trúc lưới ngay khi nhìn đúng góc (3D). Muốn đoán trúng xổ số dùng CSPRNG"
        "\n   thật, phải biết trạng thái nội bộ/seed - chuyện đó là 'tà tu' (hacker),"
        "\n   không phải 'chính đạo' (người chơi bình thường)."
    )

if __name__ == "__main__":
    main()
