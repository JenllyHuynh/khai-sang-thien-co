import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.tran_phap_1_monte_carlo.aggregate import aggregate_stats
from src.tran_phap_1_monte_carlo.lottery_config import BiasConfig, LUCKY_NUMBERS
from src.tran_phap_2_overfit.simulate import run_simulation
from src.tran_phap_2_overfit.stats import summarize as summarize_b
from src.tran_phap_5_near_miss.config import CACH_GAN_DIEM, ThamSo
from src.tran_phap_5_near_miss.simulate import chay_thi_nghiem
from src.tran_phap_5_near_miss.stats import summarize as summarize_c

CONFIG = Path(__file__).with_name("frozen_config.json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def stats(values):
    a = np.asarray(values, dtype=float)
    assert a.size > 1 and np.isfinite(a).all()
    return {"n": int(a.size), "mean": float(a.mean()),
            "sd": float(a.std(ddof=1)), "min": float(a.min()), "max": float(a.max())}


def same_numeric(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same_numeric(a[k], b[k])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            same_numeric(x, y)
    elif isinstance(a, (int, float)):
        assert np.isclose(a, b, rtol=1e-11, atol=1e-12), (a, b)
    else:
        assert a == b, (a, b)


def source_paths():
    paths = [ROOT / "src/__init__.py", ROOT / "requirements.txt", CONFIG, Path(__file__)]
    for module in ("tran_phap_1_monte_carlo", "tran_phap_2_overfit", "tran_phap_5_near_miss"):
        paths.extend(sorted((ROOT / "src" / module).glob("*.py")))
    return paths


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def environment():
    versions = {p: importlib.metadata.version(p)
                for p in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "pytest")}
    cpu = platform.processor()
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except (ImportError, OSError):
        pass
    return {"python": sys.version, "platform": platform.platform(),
            "cpu": cpu, "dependencies": versions,
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "revision_note": "Git HEAD plus archived source SHA256; working tree need not be committed."}


def audit_a(cfg, out):
    a = cfg["A"]
    df = pd.read_csv(ROOT / a["csv"])
    assert len(df) == 100 and df["seed"].tolist() == a["seeds"]
    assert df["run_index"].tolist() == list(range(100))
    assert (df["n_players_per_group"] == a["n_players_per_group"]).all()
    assert BiasConfig().__dict__ == a["weights"]
    assert sorted(LUCKY_NUMBERS) == a["lucky_numbers"]
    summary = aggregate_stats(df)
    same_numeric(summary, json.loads((ROOT / a["summary"]).read_text(encoding="utf-8")))
    assert summary["total_players_simulated_per_group"] == 500_000_000
    # Historical sources are archived verbatim, never regenerated or overwritten.
    for key in ("csv", "summary", "single_example"):
        src = ROOT / a[key]
        shutil.copy2(src, out / "legacy_A" / src.name)
    write_json(out / "A_audit.json", {"status": "passed", "aggregate_recomputed": summary,
        "caveat": "Historical environment, exact source revision and chunk size were not recorded. "
                  "Current source snapshot is not asserted to be the historical source."})
    return summary


def run_b(cfg, out):
    b = cfg["B"]
    rows = []
    ref = None
    for seed in b["seeds"] + [b["reference_seed"]]:
        start = time.perf_counter()
        result = summarize_b(run_simulation(
            n_draws=b["n_draws"], window=b["window"], test_frac=b["test_frac"], seed=seed))
        assert result["n_train_draws"] == 1357 and result["n_test_draws"] == 151
        write_json(out / "B" / f"seed_{seed:02d}.json", {"seed": seed, **result})
        if seed == b["reference_seed"]:
            ref = result
        else:
            for model, splits in result["models"].items():
                for split, metrics in splits.items():
                    rows.append({"seed": seed, "model": model, "split": split, **metrics})
        print(f"B seed {seed}: {time.perf_counter()-start:.2f}s", flush=True)
    write_csv(out / "B_runs.csv", rows)
    agg = {}
    for model in ref["models"]:
        agg[model] = {}
        for split in ("train", "test"):
            selected = [r for r in rows if r["model"] == model and r["split"] == split]
            assert len(selected) == 30
            agg[model][split] = {metric: stats([r[metric] for r in selected])
                                for metric in ref["models"][model][split]}
    legacy = json.loads((ROOT / b["legacy_reference"]).read_text(encoding="utf-8"))
    # A library-version difference may alter a legacy reference: report, never hide it.
    try:
        same_numeric(ref, legacy)
        comparison = "matches"
    except AssertionError:
        comparison = "differs; use newly archived seed-42 run for the frozen study"
    return {"n_runs": 30, "seeds": b["seeds"], "models": agg,
            "reference_42_legacy_comparison": comparison}


def run_c(cfg, out):
    c = cfg["C"]
    assert list(CACH_GAN_DIEM) == c["score_rules"]
    ts = ThamSo(**c["parameters"], bang_thuong={int(k): v for k, v in c["payouts"].items()})
    rows = []
    for seed in c["seeds"] + [c["reference_seed"]]:
        start = time.perf_counter()
        reference = seed == c["reference_seed"]
        kq = chay_thi_nghiem(
            n_agents=c["n_agents"], tham_so=ts, beta=c["beta"], seed=seed,
            n_boot=c["n_boot"], betas_quet=c["beta_sweep"] if reference else None,
            rhos_do_ben=c["rho_grid"] if reference else None)
        assert (kq["_nh"]["so_ky"] >= kq["_dc"]["so_ky"]).all()
        assert kq["so_sanh"]["chenh_ky_tb_nhom_khong_gap"] == 0
        result = summarize_c(kq)
        write_json(out / "C" / f"seed_{seed:02d}.json", result)
        if not reference:
            rows.append({"seed": seed, **{k: v for k, v in result["so_sanh"].items()
                                         if isinstance(v, (int, float))}})
        else:
            legacy = json.loads((ROOT / c["legacy_reference"]).read_text(encoding="utf-8"))
            same_numeric(result, legacy)
            assert result["quet_beta"][0]["chenh_ky_tb"] == 0
        del kq
        print(f"C seed {seed}: {time.perf_counter()-start:.2f}s", flush=True)
    write_csv(out / "C_runs.csv", rows)
    return {"n_runs": 10, "seeds": c["seeds"],
            "metrics": {k: stats([r[k] for r in rows]) for k in rows[0] if k != "seed"},
            "reference_42_legacy_comparison": "matches including sweep and sensitivity grid"}


def verify(out, check_working_tree=False):
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    for rel, expected in manifest["source_sha256"].items():
        assert digest(out / "source_snapshot" / rel) == expected, f"Archived source changed: {rel}"
        if check_working_tree:
            assert digest(ROOT / rel) == expected, f"Working-tree source changed: {rel}"
    for rel, expected in manifest["output_sha256"].items():
        assert digest(out / rel) == expected, f"Output changed: {rel}"
    for rel, expected in manifest["legacy_input_sha256"].items():
        assert digest(ROOT / rel) == expected, f"Legacy evidence changed: {rel}"
    print("Verified archived source/configuration, legacy inputs and output hashes.", flush=True)
    if not check_working_tree:
        print("Current working-tree source is not compared with the historical snapshot. "
              "Use --check-working-tree to require an exact match.", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="outputs/paper_frozen_v1")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--check-working-tree", action="store_true",
                   help="Also require current source to match the archived paths and hashes exactly.")
    args = p.parse_args()
    out = ROOT / args.output_dir
    if args.verify_only:
        verify(out, check_working_tree=args.check_working_tree)
        return
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}. Use --verify-only or a new --output-dir.")
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    out.mkdir(parents=True)
    (out / "legacy_A").mkdir()
    paths = source_paths()
    hashes = {p.relative_to(ROOT).as_posix(): digest(p) for p in paths}
    for path in paths:
        dest = out / "source_snapshot" / path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    legacy = [cfg["A"][k] for k in ("csv", "summary", "single_example")]
    legacy += [cfg["B"]["legacy_reference"], cfg["C"]["legacy_reference"]]
    manifest = {"status": "running", "config": cfg, "environment": environment(),
                "frozen_at": datetime.now(timezone(timedelta(hours=7))).isoformat(),
                "source_sha256": hashes,
                "legacy_input_sha256": {s: digest(ROOT / s) for s in legacy},
                "command": ".venv/Scripts/python.exe paper/experiments/run_frozen.py"}
    write_json(out / "manifest.json", manifest)
    start = time.perf_counter()
    a = audit_a(cfg, out)
    print("A: verified 100 legacy rows, seeds 0..99 and historical aggregate.", flush=True)
    b = run_b(cfg, out)
    c = run_c(cfg, out)
    write_json(out / "aggregate.json", {"A": a, "B": b, "C": c, "reporting": cfg["reporting"]})
    for rel, sha in hashes.items():
        assert digest(ROOT / rel) == sha, f"Source changed during run: {rel}"
    manifest["status"] = "complete"
    manifest["elapsed_seconds"] = time.perf_counter() - start
    manifest["output_sha256"] = {
        path.relative_to(out).as_posix(): digest(path)
        for path in sorted(out.rglob("*"))
        if path.is_file() and path.name != "manifest.json"}
    write_json(out / "manifest.json", manifest)
    verify(out, check_working_tree=True)


if __name__ == "__main__":
    main()
