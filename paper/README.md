# Paper reproduction guide

## 1. Three experiments

| Experiment | Vietnamese module | Purpose |
| --- | --- | --- |
| A | `src/tran_phap_1_monte_carlo/` | Uniform versus birthday/lucky-weighted ticket concentration; 100 archived populations per policy. |
| B | `src/tran_phap_2_overfit/` | Memorizing versus restricted trees on 30 chronological histories, plus the 52-draw recent-frequency supplement. |
| C | `src/tran_phap_5_near_miss/` | Paired control/sensitive agents; ten replicated populations and a separate seed-42 sensitivity analysis. |

Modules 3 and 4 are exploratory. Original Vietnamese names are retained in
code and JSON. Historical snapshots and manifests are preserved unchanged.
The manuscript LaTeX sources are maintained separately from this code repository.

## 2. Tests, verification, and reproduction

Run these commands from the `khai-sang-thien-co` repository root, after
`python -m pip install -r requirements.txt`:

```bash
python -m pytest -q

# Verify archived bytes without rerunning simulations.
python paper/experiments/run_frozen.py --verify-only

# Audit historical A inputs and rerun B/C into a NEW directory.
python paper/experiments/run_frozen.py --output-dir outputs/paper_reproduced_v1

# Reproduce the recent-frequency supplement into a NEW directory.
python paper/experiments/run_recent_frequency.py --output-dir outputs/paper_recent_frequency_reproduced_v1

# Regenerate figures using the archived supplement.
python paper/experiments/make_figures.py --heuristic-data outputs/paper_recent_frequency_v1/B_runs.csv
```

Both runners refuse to overwrite existing result directories. Choose a new
`--output-dir` for subsequent runs. Figures are written to `paper/figures/`.
The main runner reuses and audits A's historical evidence; it does not
regenerate all 100 ticket populations.

**Known verification issue:** the current checkout has 20 LF/CRLF-only
checksum mismatches among 113 archived source/output entries, so the strict
`--verify-only` command currently fails. Line-ending-normalized comparison
found no other content differences; this does not constitute a passing
byte-level audit. Frozen manifests and files have not been rewritten.
`--check-working-tree` additionally compares current code with the historical
snapshot and is not expected to pass after refactoring.

Recorded versions and source hashes are in the manifests. The historical
HEAD `d428e08` is the base of the frozen run, not a release identifier for
the current repository; its executed source is identified by the snapshot
and checksums. The recent-frequency supplement was added after inspection
of the frozen tree results. Keep the six evidence/reference directories
listed in `outputs/README.md` at their existing paths.

## 3. Glossary

| Code/JSON key | English meaning |
| --- | --- |
| `vo_vi` | Uniform ticket-selection policy |
| `tinh_cam` | Birthday/lucky-weighted ticket-selection policy |
| `tuyen_tinh` | Linear near-miss scoring rule |
| `so_ky` | Rounds played |
| `so_trung` / `so_khop` | Number of matches (overlap) |
| `doi_chung` | Control agent/group |
| `nhay` | Near-miss-sensitive agent/group |
| `thua_rong_tb` | Mean net loss (expenditure minus prizes) |
