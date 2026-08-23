# Tổng hợp kết quả mô phỏng Trận Pháp 2 thành dict gọn để in/lưu JSON
def summarize(sim_result: dict) -> dict:
    summary = {
        "n_draws": sim_result["n_draws"],
        "window": sim_result["window"],
        "n_train_draws": sim_result["n_train_draws"],
        "n_test_draws": sim_result["n_test_draws"],
        "theoretical_random_matches": sim_result["theoretical_random_matches"],
        "models": {},
    }
    for name, splits in sim_result["results"].items():
        model_summary = {}
        for split_name, df in splits.items():
            avg_matches = float(df["so_khop"].mean())
            avg_confidence = float(df["do_tin_binh_quan"].mean())
            real_accuracy = avg_matches / 6  # tỷ lệ THỰC SỰ đúng trong 6 số đã chọn
            model_summary[split_name] = {
                "avg_matches": avg_matches,
                "avg_confidence": avg_confidence,
                "real_accuracy": real_accuracy,
                "ao_tuong_gap": avg_confidence - real_accuracy,  # độ tự tin - sự thật (càng lớn càng "ảo tưởng")
                "pct_draws_5_or_6_matches": float((df["so_khop"] >= 5).mean() * 100),
            }
        summary["models"][name] = model_summary
    return summary
