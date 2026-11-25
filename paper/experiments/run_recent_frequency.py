import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.tran_phap_2_overfit.data_generator import generate_draw_history
from src.tran_phap_2_overfit.features import build_feature_table


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_recent(draws, window, start):
    # Rank past frequencies; ascending number breaks ties. No model fitting
    features = build_feature_table(draws, window=window)
    rows = []
    for t, group in features[features.draw_index >= start].groupby("draw_index"):
        selected = group.nlargest(6, "freq_last_window")["number"].to_numpy()
        # Independent direct-window check guards indexing and tie-breaking.
        counts = np.bincount(draws[t - window:t].ravel(), minlength=56)[1:]
        expected = np.argsort(-counts, kind="stable")[:6] + 1
        np.testing.assert_array_equal(selected, expected)
        rows.append({"draw_index": int(t),
                     "selected": " ".join(map(str, selected)),
                     "overlap": int(np.isin(selected, draws[t]).sum())})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "outputs/paper_recent_frequency_v1")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    config = ROOT / "paper/experiments/frozen_config.json"
    cfg = json.loads(config.read_text(encoding="utf-8"))["B"]
    frozen = ROOT / "outputs/paper_frozen_v1"
    # Confirm the history generator and features equal the archived implementations.
    pairs = {
        "src/tran_phap_2_overfit/data_generator.py": "src/tran_phap_2_overfit_ai/data_generator.py",
        "src/tran_phap_2_overfit/features.py": "src/tran_phap_2_overfit_ai/features.py",
        "src/tran_phap_1_monte_carlo/ticket_generator.py": "src/tran_phap_1_monte_carlo/ticket_generator.py",
        "src/tran_phap_1_monte_carlo/lottery_config.py": "src/tran_phap_1_monte_carlo/lottery_config.py",
    }
    sources = {rel: ROOT / rel for rel in pairs}
    for rel, archived in pairs.items():
        if sources[rel].read_text(encoding="utf-8") != (frozen / "source_snapshot" / archived).read_text(encoding="utf-8"):
            raise ValueError(f"Generator/features changed from frozen source: {rel}")
    sources["paper/experiments/run_recent_frequency.py"] = Path(__file__).resolve()
    sources["paper/experiments/frozen_config.json"] = config
    out.mkdir(parents=True)
    for rel, path in sources.items():
        dest = out / "source_snapshot" / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)

    rows = []
    for seed in cfg["seeds"] + [cfg["reference_seed"]]:
        draws = generate_draw_history(cfg["n_draws"], seed=seed)
        start = cfg["window"] + int((cfg["n_draws"] - cfg["window"]) * (1 - cfg["test_frac"]))
        result = select_recent(draws, cfg["window"], start)
        archived = json.loads((frozen / f"B/seed_{seed:02d}.json").read_text(encoding="utf-8"))
        assert len(result) == archived["n_test_draws"] == 151
        assert start - cfg["window"] == archived["n_train_draws"] == 1357
        result.to_csv(out / f"seed_{seed:02d}.csv", index=False)
        rows.append({"seed": seed, "n_test": len(result),
                     "test_overlap": float(result.overlap.mean()),
                     "draws_sha256": hashlib.sha256(draws.tobytes()).hexdigest()})
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "B_runs.csv", index=False)
    y = runs[runs.seed.isin(cfg["seeds"])].test_overlap
    summary = {"n_histories": len(y), "mean": float(y.mean()),
               "sd": float(y.std(ddof=1)), "min": float(y.min()), "max": float(y.max()),
               "reference_seed": cfg["reference_seed"],
               "reference_overlap": float(runs[runs.seed == cfg["reference_seed"]].test_overlap.iloc[0]),
               "expected_overlap": 36 / 55}
    (out / "aggregate.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    manifest = {"status": "complete", "created_utc": datetime.now(timezone.utc).isoformat(),
                "configuration_id": "paper_recent_frequency_v1", "base_B_config": cfg,
                "selection_status": "Supplement designed after inspection of frozen tree results; not preregistered.",
                "rule": "Top six previous-52-draw frequencies; ascending number breaks ties; no fit or confidence score.",
                "reference_seed_excluded_from_aggregate": True,
                "environment": {"python": platform.python_version(),
                                **{p: importlib.metadata.version(p) for p in ("numpy", "pandas")}},
                "source_sha256": {rel: digest(path) for rel, path in sources.items()},
                "frozen_manifest_sha256": digest(frozen / "manifest.json"),
                "output_sha256": {p.relative_to(out).as_posix(): digest(p)
                                  for p in sorted(out.rglob("*")) if p.is_file()}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
