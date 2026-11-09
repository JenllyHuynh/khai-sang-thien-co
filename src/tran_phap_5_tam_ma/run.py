import argparse
import json
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from .config import ThamSo, CACH_GAN_DIEM
from .simulate import chay_thi_nghiem
from .stats import summarize
from .visualize import build_main_figure, build_sweep_figure, luu_hinh

BETAS_MAC_DINH = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
RHOS_MAC_DINH = [0.95, 0.97, 0.985, 0.995]


def build_parser() -> argparse.ArgumentParser:
    d = ThamSo()
    p = argparse.ArgumentParser(
        description="Trận Pháp 5: Tâm Ma Suýt Trúng (Behavioral Agent Simulation)"
    )
    p.add_argument("--n-agents", type=int, default=10_000, help="Số agent (mỗi agent chạy 2 bản ghép cặp).")
    p.add_argument("--beta", type=float, default=1.0, help="Độ nhạy suýt trúng của bản 'nhạy' (>=0).")
    p.add_argument("--rho", type=float, default=d.rho, help="Phần hứng thú giữ lại qua kỳ sau.")
    p.add_argument("--alpha", type=float, default=d.alpha, help="Độ nhạy với kết quả tiền bạc.")
    p.add_argument("--cach-gan-diem", choices=list(CACH_GAN_DIEM), default=d.cach_gan_diem,
                   help="Cách gán điểm suýt trúng S theo số trùng.")
    p.add_argument("--h0-mean", type=float, default=d.h0_mean)
    p.add_argument("--h0-std", type=float, default=d.h0_std)
    p.add_argument("--ngan-sach-median-ve", type=float, default=d.ngan_sach_median_ve,
                   help="Ngân sách trung vị, tính bằng số vé.")
    p.add_argument("--max-ky", type=int, default=d.max_ky, help="Giới hạn số kỳ mỗi agent.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-sweep", action="store_true", help="Bỏ qua quét β và lưới độ bền.")
    p.add_argument("--output-dir", type=str, default="outputs/tran_phap_5")
    p.add_argument("--no-timestamp", action="store_true", help="Tắt timestamp")
    return p


def main() -> None:
    args = build_parser().parse_args()

    out_dir = Path(args.output_dir)
    if not args.no_timestamp:
        out_dir = Path(f"{args.output_dir}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Lưu kết quả vào -> {out_dir}\n")
    print("  Khởi động Trận Pháp 5: Tâm Ma Suýt Trúng...\n")

    tham_so = replace(
        ThamSo(), rho=args.rho, alpha=args.alpha, cach_gan_diem=args.cach_gan_diem,
        h0_mean=args.h0_mean, h0_std=args.h0_std,
        ngan_sach_median_ve=args.ngan_sach_median_ve, max_ky=args.max_ky,
    )

    t0 = time.time()
    kq = chay_thi_nghiem(
        n_agents=args.n_agents, tham_so=tham_so, beta=args.beta, seed=args.seed,
        betas_quet=None if args.no_sweep else BETAS_MAC_DINH,
        rhos_do_ben=None if args.no_sweep else RHOS_MAC_DINH,
    )
    elapsed = time.time() - t0
    s = summarize(kq)
    ss = s["so_sanh"]

    print(f"  Mô phỏng xong trong {elapsed:.1f} giây.\n")
    print("   Kết quả (hai bản ghép cặp của cùng một agent):\n")
    print(f"  Số kỳ trung bình   : đối chứng {ss['so_ky_tb_doi_chung']:.2f} | nhạy {ss['so_ky_tb_nhay']:.2f} "
          f"| chênh {ss['chenh_ky_tb']:+.3f} (CI95 {ss['chenh_ky_tb_ci95'][0]:+.3f}..{ss['chenh_ky_tb_ci95'][1]:+.3f}) "
          f"= {ss['chenh_ky_tuong_doi_pct']:+.2f}%")
    print(f"  Tiền chi TB        : đối chứng {ss['chi_tb_doi_chung']:,.0f}đ | nhạy {ss['chi_tb_nhay']:,.0f}đ "
          f"| chênh {ss['chenh_chi_tb']:+,.0f}đ")
    print(f"  Thua ròng TB       : đối chứng {ss['thua_rong_tb_doi_chung']:,.0f}đ | nhạy {ss['thua_rong_tb_nhay']:,.0f}đ")
    print(f"  Agent chơi lâu hơn : {ss['ty_le_agent_choi_lau_hon']*100:.1f}% "
          f"(agent chơi ít hơn: {ss['ty_le_agent_choi_it_hon']*100:.1f}%, bằng 0 theo cách ghép cặp)")
    print(f"  Agent từng gặp suýt trúng: {ss['ty_le_agent_tung_gap_suyt_trung']*100:.1f}% "
          f"-> nhóm này chơi thêm TB {ss['chenh_ky_tb_nhom_tung_gap']:.2f} kỳ; "
          f"nhóm chưa gặp: {ss['chenh_ky_tb_nhom_khong_gap']:.2f} kỳ")
    print(f"  Lý do dừng (đối chứng): " + ", ".join(f"{k} {v*100:.1f}%" for k, v in ss["ly_do_dung_doi_chung"].items()))
    print(f"  Lý do dừng (nhạy)     : " + ", ".join(f"{k} {v*100:.1f}%" for k, v in ss["ly_do_dung_nhay"].items()))
    print(f"  Tiền thưởng kỳ vọng / giá vé theo bảng giả định: {s['rtp_ly_thuyet']*100:.1f}%\n")

    if kq["cau_chuyen"] is not None:
        print("  Câu chuyện một agent:")
        for dong in kq["cau_chuyen"]["cac_dong"]:
            print(f"   - {dong}")
        print()

    if kq["quet_beta"] is not None:
        print("  Quét β (kéo dài tương đối so với đối chứng):")
        for r in kq["quet_beta"]:
            print(f"     β={r['beta']:<5g} -> {r['chenh_ky_tb']:+.3f} kỳ ({r['tuong_doi_pct']:+.2f}%), "
                  f"{r['ty_le_agent_choi_lau_hon']*100:.1f}% agent chơi lâu hơn")
        print()

    luu_hinh(build_main_figure(kq), out_dir / "tam_ma_suyt_trung.png")
    print(f"  Đã lưu biểu đồ chính -> {out_dir / 'tam_ma_suyt_trung.png'}")
    fig2 = build_sweep_figure(kq["quet_beta"], kq["do_ben"])
    if fig2 is not None:
        luu_hinh(fig2, out_dir / "quet_beta_va_do_ben.png")
        print(f"  Đã lưu biểu đồ quét β -> {out_dir / 'quet_beta_va_do_ben.png'}")

    with open(out_dir / "tong_ket.json", "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2, default=str)
    print(f"  Đã lưu tổng kết JSON -> {out_dir / 'tong_ket.json'}")

    print(
        "\n   Trận Pháp 5 hoàn thành. Nhớ: chiều của hiệu ứng (nhạy suýt trúng thì chơi lâu hơn) được\n"
        "   LẬP TRÌNH sẵn; thứ đáng đọc là độ lớn và nó đổi thế nào khi giả định đổi. Đây là phát hiện\n"
        "   về mô hình, chưa phải phát hiện về con người."
    )


if __name__ == "__main__":
    main()
