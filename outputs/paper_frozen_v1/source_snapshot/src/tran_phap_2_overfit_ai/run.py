import argparse
import json
import time
from pathlib import Path
from datetime import datetime

from .simulate import run_simulation
from .stats import summarize
from .visualize import plot_overfitting_summary

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trận Pháp 2: Lão Tặc AI - Kẻ Học Vẹt (Overfitted Prophet)"
    )
    parser.add_argument(
        "--n-draws", type=int, default=1560,
        help="Số lần quay số lịch sử giả lập (mặc định 1560 ~ 10 năm x 3 lần/tuần)",
    )
    parser.add_argument(
        "--window", type=int, default=52,
        help="Cửa sổ tính tần suất/khoảng cách gần đây (đơn vị: số lần quay).",
    )
    parser.add_argument(
        "--test-frac", type=float, default=0.1,
        help="Tỷ lệ số lần quay CUỐI CÙNG dùng làm 'tương lai' chưa từng thấy để kiểm tra.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/tran_phap_2")

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

    print(f"  Khởi động Trận Pháp 2 với {args.n_draws:,} lần quay lịch sử giả lập...\n")

    t0 = time.time()
    result = run_simulation(
        n_draws=args.n_draws, window=args.window, test_frac=args.test_frac, seed=args.seed
    )
    elapsed = time.time() - t0
    summary = summarize(result)

    print(f" Huấn luyện + đánh giá xong trong {elapsed:.1f} giây.\n")
    print(" Chân Lý Hé Lộ:\n")
    print(f" Đoán mò lý thuyết (random 6/55): trung bình {summary['theoretical_random_matches']:.3f} số trúng / 6 số\n")

    for name, splits in summary["models"].items():
        print(f"  A.I {name}:")
        print(
            f"     - Trên dữ liệu ĐÃ HỌC ({summary['n_train_draws']} lần quay): "
            f"trung bình {splits['train']['avg_matches']:.3f} số trúng, "
            f"độ tự tin {splits['train']['avg_confidence']*100:.1f}%"
        )
        print(
            f"     - Trên TƯƠNG LAI chưa thấy ({summary['n_test_draws']} lần quay): "
            f"trung bình {splits['test']['avg_matches']:.3f} số trúng, "
            f"độ tự tin {splits['test']['avg_confidence']*100:.1f}% "
            f"(  khoảng cách ảo tưởng: {splits['test']['ao_tuong_gap']*100:+.1f} điểm %)"
        )
        print()

    plot_path = out_dir / "hoc_vet_vs_thuc_te.png"
    plot_overfitting_summary(summary, plot_path)
    print(f"  Đã lưu biểu đồ -> {plot_path}")

    summary_path = out_dir / "tong_ket.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f" Đã lưu tổng kết JSON -> {summary_path}")

    print(
        "\n   Trận Pháp 2 hoàn thành! Ngộ ra: một mô hình có thể 'học thuộc lòng' quá khứ"
        "\n   hoàn hảo tuyệt đối, mà vẫn vô dụng hoàn toàn trước tương lai - nhất là khi"
        "\n   tương lai đó là random thuần túy, không hề có quy luật thật để học."
    )

if __name__ == "__main__":
    main()
