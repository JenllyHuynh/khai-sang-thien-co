**English** | [Tiếng Việt](README.vi.md)

# Khai Sang Thien Co

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24526868.svg)](https://doi.org/10.5281/zenodo.24526868)

A reproducible simulation framework for studying ticket concentration,
predictive overfitting, and near-miss persistence in an idealized fair 6/55
lottery. This repository accompanies the preprint [*Khai Sang Thien Co:
A Reproducible Simulation Study of Ticket Concentration, Predictive
Overfitting, and Near-Miss Mechanisms in a 6/55 Lottery*](https://doi.org/10.5281/zenodo.24526868)
by Kim-Hue Huynh.

The draw law is fixed: every valid six-number combination has the same
probability. The experiments examine how ticket choices, historical
prediction rules, and assumed behavioral responses affect different outcomes
around that law. They are computational comparisons, not measurements of
human behavior or tests of an operating lottery.

## Experiments

| Paper experiment | Source module | Question |
| --- | --- | --- |
| A: Ticket concentration | [`tran_phap_1_monte_carlo`](src/tran_phap_1_monte_carlo/) | How do birthday/lucky-number weights change duplicate-ticket exposure? |
| B: Historical prediction | [`tran_phap_2_overfit`](src/tran_phap_2_overfit/) | How do memorizing and restricted trees generalize to future draws? A separate recent-frequency baseline ranks the previous 52 draws. |
| C: Near-miss persistence | [`tran_phap_5_near_miss`](src/tran_phap_5_near_miss/) | How much does an assumed positive partial-match response extend paired agents' participation? |

Modules 3 (chaos/PRNG demonstrations) and 4 (synthetic luck/narrative
comparisons) are exploratory and outside the paper's three main experiments.
The original Vietnamese module names and JSON keys are retained; see the
[paper guide and glossary](paper/README.md).

## Getting started

Use Python 3.12 for the recorded experiment environment. From the repository
root, install the dependencies and run the tests:

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

The requirements specify supported ranges, not an exact environment lock.
The manifests record the versions used for the archived results.

For the optional interactive dashboard:

```bash
streamlit run dashboard/app.py
```

The simulation and tests do not require an API key. The dashboard's optional
Statistical Adversary uses the Gemini API; setup is documented in the
[Vietnamese guide](README.vi.md#statistical-adversary---đối-thủ-phản-biện-chạy-trên-gemini-api).

## Reproducing the paper

Follow the [three-experiment reproduction guide](paper/README.md) for the test,
integrity-check, and reproduction commands. The
[outputs index](outputs/README.md) distinguishes frozen evidence, historical
reference inputs, and exploratory runs. Generated PDF/PNG figures are in
[`paper/figures/`](paper/figures/).

The frozen manifest records historical HEAD
`d428e08bc6e146ac067be06c111351a6ca974f14` plus the executed source snapshot
and SHA256 checksums. That HEAD alone does not identify the full executed
working tree. Later cleanup, documentation, and supplements do not change
the provenance of the frozen results.

Byte-level verification currently detects LF/CRLF differences in the frozen
checkout. This is a known failure, not a successful integrity check; see the
[verification note](paper/experiments/README.md#kiểm-tra-bộ-kết-quả-đã-đóng-băng).

## Citation and license

If you reference the manuscript or this simulation framework, please cite the
Zenodo preprint (also provided in [`CITATION.cff`](CITATION.cff)):

```bibtex
@misc{huynh2026khaisangthienco,
  author       = {Huynh, Kim-Hue},
  title        = {Khai Sang Thien Co: A Reproducible Simulation Study of Ticket Concentration, Predictive Overfitting, and Near-Miss Mechanisms in a 6/55 Lottery},
  month        = nov,
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.24526868},
  url          = {https://doi.org/10.5281/zenodo.24526868}
```

Released under the [MIT License](LICENSE). The full Vietnamese introduction,
including its original martial-arts storytelling, is preserved in
[`README.vi.md`](README.vi.md).
