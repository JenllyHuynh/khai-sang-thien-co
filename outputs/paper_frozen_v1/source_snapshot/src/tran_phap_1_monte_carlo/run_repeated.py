import argparse
import csv
import datetime
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np

from .aggregate import aggregate_stats, load_runs
from .lottery_config import N_NUMBERS, PICK
from .simulate import generate_ids_in_chunks
from .stats import cluster_stats
from .ticket_generator import generate_bias, generate_uniform
from .visualize import plot_repeated_runs

CSV_COLUMNS = [
    "run_index", "seed", "n_players_per_group", "elapsed_seconds",
    "vo_vi_n_unique", "vo_vi_unique_ratio", "vo_vi_pct_sharing", "vo_vi_max_cluster",
    "tinh_cam_n_unique", "tinh_cam_unique_ratio", "tinh_cam_pct_sharing", "tinh_cam_max_cluster",
]

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Trận Pháp 1 - chạy lặp nhiều trial độc lập, gộp kết quả vào 1 CSV duy nhất."
    )
    parser.add_argument(
        "--n-players-per-run", type=int, default=5_000_000,
        help="Số người chơi / phe MỖI LẦN CHẠY (mặc định 5 triệu - khuyến nghị khi sẽ "
             "chạy lặp hàng trăm lần trên máy 8GB RAM; nếu chỉ chạy 1 lần, "
             "có thể dùng con số lớn hơn thoải mái, xem run.py).",
    )
    parser.add_argument("--n-runs", type=int, default=100, help="Số lần chạy muốn thêm vào lần này.")
    parser.add_argument("--chunk-size", type=int, default=200_000, help="Kích thước lô sinh vé nội bộ mỗi lần chạy.")
    parser.add_argument("--seed-start", type=int, default=0, help="Seed gốc - mỗi lần chạy dùng seed_start + run_index.")
    parser.add_argument("--output-dir", type=str, default="outputs/tran_phap_1_repeated")
    parser.add_argument("--csv-name", type=str, default="ket_qua_theo_lan_chay.csv")
    parser.add_argument(
        "--fresh", action="store_true",
        help="Bỏ hết kết quả cũ (nếu có), bắt đầu lại từ lần chạy #0.",
    )
    parser.add_argument(
        "--aggregate-only", action="store_true",
        help="Không chạy sim mới - chỉ đọc CSV hiện có và xuất lại báo cáo tổng hợp (PNG + JSON).",
    )
    parser.add_argument(
        "--no-timestamp", action="store_true",
        help="Tắt timestamp"
    )
    return parser

def _existing_run_count(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    with open(csv_path, "r", encoding="utf-8") as f:
        return max(0, sum(1 for _ in f) - 1)  # trừ dòng header

def _run_one_trial(n_players: int, chunk_size: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)

    ids_vo_vi = generate_ids_in_chunks(n_players, chunk_size, generate_uniform, rng, verbose=False)
    ids_tinh_cam = generate_ids_in_chunks(n_players, chunk_size, generate_bias, rng, verbose=False)

    stats_vo_vi = cluster_stats(ids_vo_vi)
    stats_tinh_cam = cluster_stats(ids_tinh_cam)
    del ids_vo_vi, ids_tinh_cam  # giải phóng ngay, không kéo dài qua vòng lặp tiếp theo

    return {
        "vo_vi_n_unique": stats_vo_vi["n_unique_combos"],
        "vo_vi_unique_ratio": stats_vo_vi["unique_ratio"],
        "vo_vi_pct_sharing": stats_vo_vi["pct_players_sharing_ticket"],
        "vo_vi_max_cluster": stats_vo_vi["max_cluster_size"],
        "tinh_cam_n_unique": stats_tinh_cam["n_unique_combos"],
        "tinh_cam_unique_ratio": stats_tinh_cam["unique_ratio"],
        "tinh_cam_pct_sharing": stats_tinh_cam["pct_players_sharing_ticket"],
        "tinh_cam_max_cluster": stats_tinh_cam["max_cluster_size"],
    }

def _write_aggregate_report(csv_path: Path, out_dir: Path) -> None:
    df = load_runs(csv_path)
    agg = aggregate_stats(df)

    plot_path = out_dir / "tong_hop_nhieu_lan_chay.png"
    plot_repeated_runs(df, agg, plot_path)

    summary_path = out_dir / "tong_hop_nhieu_lan_chay.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2, default=str)

    print("\n  Chân Lý Hé Lộ (tổng hợp qua tất cả các lần chạy):")
    print(f"  Tổng số lần chạy: {agg['n_runs']:,}")
    print(f"  Tổng số vé đã mô phỏng / phe: {agg['total_players_simulated_per_group']:,}")
    print(f"\n    Phe Vô Vi: trung bình {agg['vo_vi']['pct_sharing_weighted_mean']:.4f}% trùng vé "
          f"(độ lệch chuẩn giữa các lần chạy: {agg['vo_vi']['pct_sharing_std_across_runs']:.4f}, "
          f"dao động {agg['vo_vi']['pct_sharing_min']:.4f}%–{agg['vo_vi']['pct_sharing_max']:.4f}%)")
    print(f"    Phe Tình Cảm: trung bình {agg['tinh_cam']['pct_sharing_weighted_mean']:.4f}% trùng vé "
          f"(độ lệch chuẩn giữa các lần chạy: {agg['tinh_cam']['pct_sharing_std_across_runs']:.4f}, "
          f"dao động {agg['tinh_cam']['pct_sharing_min']:.4f}%–{agg['tinh_cam']['pct_sharing_max']:.4f}%)")
    print(f"\n    Chênh lệch trung bình: Phe Tình Cảm trùng vé cao gấp "
          f"~{agg['ty_le_chenh_lech_trung_binh']:.1f} lần Phe Vô Vi - ổn định qua {agg['n_runs']} lần thử độc lập.")
    print(f"\n   Biểu đồ tổng hợp -> {plot_path}")
    print(f"  JSON tổng hợp    -> {summary_path}")

def main() -> None:
    args = build_parser().parse_args()

    if args.no_timestamp:
        out_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"{args.output_dir}_{timestamp}")

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / args.csv_name
    print(f"  Lưu kết quả vào -> {out_dir}\n")

    if args.fresh and csv_path.exists():
        csv_path.unlink()
        print(f"  Đã xóa kết quả cũ tại {csv_path}, bắt đầu lại từ đầu.\n")

    if args.aggregate_only:
        if not csv_path.exists():
            print(f" Không tìm thấy {csv_path} - chưa có lần chạy nào để tổng hợp.")
            return
        _write_aggregate_report(csv_path, out_dir)
        return

    start_run = _existing_run_count(csv_path)
    if start_run > 0:
        print(f"  Tìm thấy {start_run} lần chạy đã có tại {csv_path}, chạy tiếp từ lần #{start_run}.\n")

    write_header = not csv_path.exists()
    total_players_this_session_estimate = args.n_players_per_run * args.n_runs
    print(
        f"  Chuẩn bị chạy {args.n_runs} lần × {args.n_players_per_run:,} vé/phe/lần "
        f"(~{total_players_this_session_estimate:,} vé/phe cho phiên này)...\n"
    )

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
            f.flush()

        session_t0 = time.time()
        for i in range(args.n_runs):
            run_index = start_run + i
            seed = args.seed_start + run_index

            t0 = time.time()
            trial = _run_one_trial(args.n_players_per_run, args.chunk_size, seed)
            elapsed = time.time() - t0

            row = {
                "run_index": run_index,
                "seed": seed,
                "n_players_per_group": args.n_players_per_run,
                "elapsed_seconds": round(elapsed, 3),
                **trial,
            }
            writer.writerow(row)
            f.flush()  # ghi ngay xuống đĩa - mất điện/Ctrl+C giữa chừng cũng không mất dữ liệu

            print(
                f"  [{run_index + 1}/{start_run + args.n_runs}] {elapsed:.1f}s "
                f"| Vô Vi trùng {trial['vo_vi_pct_sharing']:.3f}% "
                f"| Tình Cảm trùng {trial['tinh_cam_pct_sharing']:.3f}%"
            )

        session_elapsed = time.time() - session_t0

    print(f"\n  Phiên này chạy xong {args.n_runs} lần trong {session_elapsed/60:.1f} phút.")
    print(f" Kết quả từng lần chạy -> {csv_path}")

    _write_aggregate_report(csv_path, out_dir)

if __name__ == "__main__":
    main()
