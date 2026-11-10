import json
import hashlib
import sys
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
from src.tran_phap_5_near_miss.config import ThamSo
from src.tran_phap_5_near_miss.draws import pmf_so_trung
from src.tran_phap_5_near_miss.dynamics import bang_diem_suyt_trung
from src.tran_phap_5_near_miss.simulate import tao_hat_giong, chay_cap
from src.tran_phap_5_near_miss.stats import dem_suyt_trung

config_path = ROOT / "paper/experiments/frozen_config.json"
cfg = json.loads(config_path.read_text(encoding="utf-8"))["C"]
data_dir = Path(__file__).with_name("paper_data")
reference = json.loads((data_dir / "C_ref.json").read_text(encoding="utf-8"))
frozen_reference = json.loads((ROOT / "outputs/paper_frozen_v1/C/seed_42.json")
                              .read_text(encoding="utf-8"))
for export_key, frozen_key in (("ref_so_sanh", "so_sanh"),
                               ("quet_beta", "quet_beta"), ("do_ben", "do_ben")):
    assert reference[export_key] == frozen_reference[frozen_key]
# Check that every figure replicate agrees with the archived frozen summaries.
names = {"unres": "Lão Tặc AI (Học Vẹt)", "res": "Đạo Sĩ Khiêm Tốn (Baseline)"}
with (data_dir / "B_runs.csv").open(encoding="utf-8") as f:
    b_rows = list(csv.DictReader(f))
assert sorted(int(r["seed"]) for r in b_rows) == list(range(30)) + [42]
for row in b_rows:
    seed = int(row["seed"])
    saved = json.loads((ROOT / f"outputs/paper_frozen_v1/B/seed_{seed:02d}.json")
                       .read_text(encoding="utf-8"))
    for tag, name in names.items():
        for split in ("train", "test"):
            for suffix, metric in (("overlap", "avg_matches"), ("score", "avg_confidence"),
                                   ("gap", "ao_tuong_gap")):
                assert np.isclose(float(row[f"{tag}_{split}_{suffix}"]),
                                  saved["models"][name][split][metric], atol=1e-12, rtol=1e-11)
with (data_dir / "C_runs.csv").open(encoding="utf-8") as f:
    c_rows = list(csv.DictReader(f))
assert sorted(int(r["seed"]) for r in c_rows) == cfg["seeds"]
for row in c_rows:
    seed = int(row["seed"])
    saved = json.loads((ROOT / f"outputs/paper_frozen_v1/C/seed_{seed:02d}.json")
                       .read_text(encoding="utf-8"))["so_sanh"]
    for col, key, factor in (
        ("ctrl", "so_ky_tb_doi_chung", 1), ("sens", "so_ky_tb_nhay", 1),
        ("extra", "chenh_ky_tb", 1), ("rel_pct", "chenh_ky_tuong_doi_pct", 1),
        ("pct_longer", "ty_le_agent_choi_lau_hon", 100),
        ("pct_exposed", "ty_le_agent_tung_gap_suyt_trung", 100)):
        assert np.isclose(float(row[col]), saved[key] * factor, atol=1e-12, rtol=1e-11)
ts = ThamSo(**cfg["parameters"],
            bang_thuong={int(k): v for k, v in cfg["payouts"].items()})
rows = []
totals = {g: {"n": 0, "sum_extra": 0, "sum_control": 0, "sum_sensitive": 0}
          for g in ("all", "exposed", "extended", "exposed_unextended", "unexposed")}
for seed in cfg["seeds"]:
    hat = tao_hat_giong(cfg["n_agents"], ts, seed)
    dc, nh = chay_cap(hat, ts, beta=cfg["beta"])
    d = nh["so_ky"] - dc["so_ky"]
    assert (d >= 0).all()
    exposed = dem_suyt_trung(hat["so_trung"], nh["so_ky"],
                             bang_diem=bang_diem_suyt_trung(ts.cach_gan_diem)) > 0
    extended = d > 0
    assert not (extended & ~exposed).any()
    saved = json.loads((ROOT / f"outputs/paper_frozen_v1/C/seed_{seed:02d}.json")
                       .read_text(encoding="utf-8"))["so_sanh"]
    assert np.isclose(d.mean(), saved["chenh_ky_tb"])
    assert np.isclose(exposed.mean(), saved["ty_le_agent_tung_gap_suyt_trung"])
    assert np.isclose(extended.mean(), saved["ty_le_agent_choi_lau_hon"])
    row = {"seed": seed}
    masks = {"all": np.ones(len(d), dtype=bool), "exposed": exposed,
             "extended": extended, "exposed_unextended": exposed & ~extended,
             "unexposed": ~exposed}
    for group, mask in masks.items():
        values = {"n": int(mask.sum()), "sum_extra": int(d[mask].sum()),
                  "sum_control": int(dc["so_ky"][mask].sum()),
                  "sum_sensitive": int(nh["so_ky"][mask].sum())}
        for key, value in values.items():
            totals[group][key] += value
        row[group] = {**values, "mean_extra": float(d[mask].mean()),
                      "mean_control": float(dc["so_ky"][mask].mean())}
    rows.append(row)
for group, values in totals.items():
    values["fraction"] = values["n"] / totals["all"]["n"]
    values["mean_extra"] = values["sum_extra"] / values["n"]
    values["mean_control"] = values["sum_control"] / values["n"]
    values["mean_sensitive"] = values["sum_sensitive"] / values["n"]
    values["relative_increase_pct"] = 100 * values["sum_extra"] / values["sum_control"]
out = {"configuration": "paper_frozen_v1", "seeds": cfg["seeds"],
       "figure_replication_inputs": "B and C CSV values verified against all frozen per-seed summaries",
       "figure_input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(data_dir.iterdir()) if p.name in
                               ("A_summary.json", "B_runs.csv", "C_runs.csv", "C_ref.json")},
       "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
       "pmf_near_miss_per_ticket": float(pmf_so_trung()[3:6].sum()),
       "aggregation": "Descriptive subgroup totals across ten equally sized populations; "
                      "not a causal subgroup estimate or a new independent population.",
       "pooled_descriptive_groups": totals, "per_population": rows}
dest = Path(__file__).with_name("paper_data") / "C_subgroups.json"
dest.write_text(json.dumps(out, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps({k: v for k, v in out.items() if k != "per_population"}, indent=2))
