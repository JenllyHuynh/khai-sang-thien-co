import argparse
import json
import time
from  datetime import datetime
from pathlib import Path

from .encode import decode_ticket
from .simulate import run_simulation
from .stats import cluster_stats, top_popular_combos
from .visualize import plot_summary

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trận Pháp 1: Vạn Kiếp Quy Tông - mô phỏng Monte Carlo xổ số 6/55"
    )
    parser.add_argument(
        "--n-players", type=int, default=2_000_000,
        help="Số người chơi mô phỏng MỖI PHE (mặc định 2 triệu - hợp với máy 8GB RAM). "
             "Bí kíp gốc nói '1 tỷ' nhưng con số đó sẽ khiến máy yếu ngồi chờ đến tắt thở; "
             "2 triệu/phe đã đủ để thấy rõ chân lý mà chạy trong vài chục giây.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=200_000,
        help="Số vé sinh mỗi lô (giảm xuống 50,000-100,000 nếu máy vẫn đuối RAM).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed để tái lập kết quả.")
    parser.add_argument(
        "--output-dir", type=str, default="outputs/tran_phap_1",
        help="Thư mục lưu biểu đồ + báo cáo.",
    )
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
    print(f" Lưu kết quả vào -> {out_dir}\n")

    print(f"  Khởi động Trận Pháp 1 với {args.n_players:,} người chơi / phe "
          f"(lô {args.chunk_size:,} vé)...\n")

    t0 = time.time()
    result = run_simulation(n_players=args.n_players, chunk_size=args.chunk_size, seed=args.seed)
    elapsed = time.time() - t0

    stats_vo_vi = cluster_stats(result["ids_vo_vi"])
    stats_tinh_cam = cluster_stats(result["ids_tinh_cam"])

    print(f"\n Sinh vé xong trong {elapsed:.1f} giây.")
    print("\n Chân Lý Hé Lộ:")
    print(
        f"  Xác suất trúng Jackpot lý thuyết (như nhau cho CẢ HAI phe): "
        f"1 / {result['total_combos']:,} ≈ {result['theoretical_win_probability']:.3e}"
    )

    print("\n    Phe Vô Vi (random thuần túy):")
    print(f"     - Tổ hợp độc nhất: {stats_vo_vi['n_unique_combos']:,} / {stats_vo_vi['n_players']:,} vé "
          f"({stats_vo_vi['unique_ratio']*100:.4f}% không trùng ai)")
    print(f"     - % người chơi bị trùng vé với ai đó: {stats_vo_vi['pct_players_sharing_ticket']:.4f}%")
    print(f"     - Cụm trùng đông nhất: {stats_vo_vi['max_cluster_size']} người cùng chọn 1 tổ hợp")

    print("\n    Phe Tình Cảm (số đẹp/ngày sinh):")
    print(f"     - Tổ hợp độc nhất: {stats_tinh_cam['n_unique_combos']:,} / {stats_tinh_cam['n_players']:,} vé "
          f"({stats_tinh_cam['unique_ratio']*100:.4f}% không trùng ai)")
    print(f"     - % người chơi bị trùng vé với ai đó: {stats_tinh_cam['pct_players_sharing_ticket']:.4f}%")
    print(f"     - Cụm trùng đông nhất: {stats_tinh_cam['max_cluster_size']} người cùng chọn 1 tổ hợp")

    ty_le_chenh_lech = (
        stats_tinh_cam["pct_players_sharing_ticket"] / stats_vo_vi["pct_players_sharing_ticket"]
        if stats_vo_vi["pct_players_sharing_ticket"] > 0 else float("inf")
    )
    print(f"\n   ️  Phe Tình Cảm có nguy cơ trùng vé cao gấp ~{ty_le_chenh_lech:.1f} lần Phe Vô Vi,"
          f" dù xác suất trúng thực tế NHƯ NHAU!")

    # Lưu top tổ hợp phổ biến nhất của phe Tình Cảm
    top_combos = top_popular_combos(stats_tinh_cam, decode_ticket, top_n=15)
    top_combos_path = out_dir / "top_to_hop_pho_bien_tinh_cam.csv"
    top_combos.to_csv(top_combos_path, index=False, encoding="utf-8-sig")
    print(f"\n Đã lưu top 15 tổ hợp phổ biến nhất (phe Tình Cảm) -> {top_combos_path}")

    # Lưu biểu đồ
    plot_path = out_dir / "so_sanh_hai_phe.png"
    plot_summary(stats_vo_vi, stats_tinh_cam, plot_path)
    print(f" Đã lưu biểu đồ so sánh -> {plot_path}")

    # Lưu tổng kết JSON
    summary = {
        "n_players_per_group": args.n_players,
        "elapsed_seconds": round(elapsed, 2),
        "theoretical_win_probability": result["theoretical_win_probability"],
        "total_combos": result["total_combos"],
        "vo_vi": {k: v for k, v in stats_vo_vi.items() if not k.startswith("_")},
        "tinh_cam": {k: v for k, v in stats_tinh_cam.items() if not k.startswith("_")},
        "ty_le_chenh_lech_nguy_co_trung_ve": (
            stats_tinh_cam["pct_players_sharing_ticket"] / stats_vo_vi["pct_players_sharing_ticket"]
            if stats_vo_vi["pct_players_sharing_ticket"] > 0 else float("inf")
        ),
    }
    summary_path = out_dir / "tong_ket.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f" Đã lưu tổng kết JSON -> {summary_path}")

    print(
        "\n   Trận Pháp 1 hoàn thành! Ngộ ra: chọn số đẹp không hề tăng xác suất trúng,"
        "\n   nhưng nếu trúng thật, khả năng phải chia sẻ giải thưởng với người khác cao hơn hẳn."
    )

if __name__ == "__main__":
    main()
