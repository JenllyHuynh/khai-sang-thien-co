import argparse
import json
import time
from pathlib import Path
from datetime import datetime

from .simulate import run_simulation
from .stats import summarize
from .visualize import plot_full_summary

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trận Pháp 4: Bàn Cờ Nhân Quả - Họa Phúc Song Hành (Quantifying Luck)"
    )
    parser.add_argument("--n-people", type=int, default=2_000_000,
                         help="Kích thước quần thể chính để dựng Bàn Cờ Nhân Quả.")
    parser.add_argument("--p-win", type=float, default=0.001,
                         help="Xác suất 'trúng jackpot' TRONG MÔ PHỎNG (phóng đại so với thực tế "
                              "để có đủ mẫu người trúng phân tích thống kê - xem README)")
    parser.add_argument("--p-hoa-baseline", type=float, default=0.05,
                         help="Xác suất nền 'gặp họa' (số minh họa cho mục đích giáo dục)")
    parser.add_argument("--n-trials-null", type=int, default=200,
                         help="Số lần chạy độc lập để dựng phân phối 'nhiễu do may rủi'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs/tran_phap_4")

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

    print("  Khởi động Trận Pháp 4: Bàn Cờ Nhân Quả...\n")

    t0 = time.time()
    result = run_simulation(
        n_people=args.n_people,
        p_win=args.p_win,
        p_hoa_baseline=args.p_hoa_baseline,
        n_trials_null=args.n_trials_null,
        seed=args.seed,
    )
    elapsed = time.time() - t0
    summary = summarize(result)

    print(f"  Mô phỏng xong trong {elapsed:.1f} giây.\n")
    print("   Chân Lý Hé Lộ:\n")
    print(f"  Quần thể chính: {args.n_people:,} người, "
          f"{summary['n_a1']:,} người 'trúng jackpot' trong mô phỏng.\n")

    print("  Bàn Cờ Nhân Quả (số người):")
    print(f"     Trúng + Gặp Họa    : {summary['table'][0][0]:,}")
    print(f"     Trúng + Không Họa  : {summary['table'][0][1]:,}")
    print(f"     Không Trúng + Họa  : {summary['table'][1][0]:,}")
    print(f"     Không Trúng + Không: {summary['table'][1][1]:,}\n")

    print(f"  P(Họa | Trúng)        = {summary['p_hoa_given_trung']*100:.4f}%")
    print(f"  P(Họa | Không Trúng)  = {summary['p_hoa_given_khong_trung']*100:.4f}%")
    print(f"  Chênh lệch            = {summary['chenh_lech_diem_phan_tram']:+.4f} điểm %")
    print(f"  Odds Ratio            = {summary['odds_ratio']:.3f}  (≈1.0 = không liên hệ)")
    print(f"  Chi² p-value          = {summary['p_value']:.4f} "
          f"({'CÓ' if summary['co_y_nghia_thong_ke'] else 'KHÔNG'} ý nghĩa thống kê ở mức 5%)\n")

    print(
        f"  Qua {args.n_trials_null} lần chạy độc lập (A,B luôn độc lập tuyệt đối theo thiết kế), "
        f"chênh lệch P(Họa|Trúng)-P(Họa|Không Trúng) dao động ngẫu nhiên quanh "
        f"{summary['null_distribution_mean']:+.4f} điểm % "
        f"(độ lệch chuẩn {summary['null_distribution_std']:.4f}, "
        f"khoảng {summary['null_distribution_min']:+.3f}% đến {summary['null_distribution_max']:+.3f}%)."
    )
    print(
        "  -> Một câu chuyện đơn lẻ rơi vào đâu đó trong khoảng này hoàn toàn có thể chỉ là"
        "\n     may rủi thống kê, không phải bằng chứng của 'lời nguyền xổ số'."
    )

    plot_path = out_dir / "ban_co_nhan_qua.png"
    plot_full_summary(result["table"], result["independence_result"], result["null_diffs"], plot_path)
    print(f"\n  Đã lưu biểu đồ -> {plot_path}")

    summary_path = out_dir / "tong_ket.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f" Đã lưu tổng kết JSON -> {summary_path}")

    print(
        "\n   Trận Pháp 4 hoàn thành! Ngộ ra: 'May mắn' và 'Xui xẻo' không phải linh khí"
        "\n   trong vũ trụ - chúng chỉ là những cái NHÃN con người dán lên các sự kiện"
        "\n   ngẫu nhiên độc lập, để tự an ủi hoặc tự hào. Trong họa có phúc, trong phúc"
        "\n   có họa - cả hai đều là chuyện của con người, không phải của Random."
    )

if __name__ == "__main__":
    main()
