import argparse, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = Path(__file__).resolve().parent

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

BLUE, ORANGE, GREY, GREEN = "#0072B2", "#D55E00", "#555555", "#009E73"
_serif = [f.name for f in font_manager.fontManager.ttflist]
plt.rcParams.update({
    "font.family": "serif", "font.serif": [n for n in ("Times New Roman", "Liberation Serif", "Nimbus Roman", "DejaVu Serif") if n in _serif] or ["serif"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
    "mathtext.fontset": "stix", "figure.dpi": 150,
})
COLW, FULLW = 3.45, 7.1


def panel(ax, tag):
    ax.text(-0.04, 1.12, tag, transform=ax.transAxes, fontweight="bold", fontsize=8, va="bottom", ha="right")


def fig_concentration(cfg, data, out):
    df = pd.read_csv(ROOT / cfg["A"]["csv"])
    an = json.load(open(Path(data) / "A_summary.json"))["analytic"]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(FULLW, 2.65))
    for ax, col, color, name, key, tag in ((axes[0], "vo_vi_pct_sharing", BLUE, "Uniform", "uniform_n5000000", "(a)"),
                                           (axes[1], "tinh_cam_pct_sharing", ORANGE, "Weighted", "weighted_n5000000", "(b)")):
        y = df[col].values; x = rng.uniform(-0.25, 0.25, len(y))
        ax.scatter(x, y, s=5, color=color, alpha=0.55, linewidths=0)
        ax.axhline(y.mean(), color=color, lw=1.0, label=f"mean {y.mean():.4f}")
        ax.axhline(an[key]["share_pct"], color="black", lw=0.8, ls="--", label=f"analytic {an[key]['share_pct']:.4f}")
        c = an[key]["share_pct"]; ax.set_ylim(c - 0.085, c + 0.125); ax.set_xlim(-0.6, 0.6); ax.set_xticks([0]); ax.set_xticklabels([name])
        ax.legend(loc="upper center", frameon=False, handlelength=1.4, borderaxespad=0.2, bbox_to_anchor=(0.5, 1.0))
        panel(ax, tag)
    axes[0].set_ylabel("Players sharing a ticket (%)")
    fig.tight_layout(w_pad=1.2)
    fig.savefig(Path(out) / "ticket_concentration.pdf", bbox_inches="tight"); fig.savefig(Path(out) / "ticket_concentration.png", bbox_inches="tight", dpi=300); plt.close(fig)


def fig_overfit(data, out):
    df = pd.read_csv(Path(data) / "B_runs.csv"); rep = df[df.seed != 42]
    rng = np.random.default_rng(1)
    fig, axes = plt.subplots(1, 3, figsize=(FULLW, 2.85), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    models = (("unres", "Unrestricted", ORANGE), ("res", "Restricted", BLUE))
    for ax, split, title, tag in ((axes[0], "train", "Training period", "(a)"), (axes[1], "test", "Chronological test period", "(b)")):
        for i, (key, name, color) in enumerate(models):
            y = rep[f"{key}_{split}_overlap"].values
            ax.scatter(i + rng.uniform(-0.18, 0.18, len(y)), y, s=7, color=color, alpha=0.6, linewidths=0)
            ax.errorbar(i, y.mean(), yerr=y.std(ddof=1), color="black", marker="_", ms=10, lw=0.9, capsize=2.5, zorder=3)
        ax.axhline(36 / 55, color=GREY, ls="--", lw=0.8, label="random reference 36/55")
        ax.set_xticks([0, 1]); ax.set_xticklabels([m[1] for m in models]); ax.set_xlim(-0.6, 1.6); ax.set_title(title); panel(ax, tag)
    axes[0].set_ylabel("Mean overlap with six drawn numbers"); axes[0].set_ylim(0, 6.5); axes[1].set_ylim(0.45, 0.85)
    axes[1].legend(loc="upper left", frameon=False, handlelength=1.6)
    ax = axes[2]; w = 0.34; xs = np.arange(2)
    score = [rep[f"{k}_test_score"].mean() for k, _, _ in models]; score_sd = [rep[f"{k}_test_score"].std(ddof=1) for k, _, _ in models]
    hit = [(rep[f"{k}_test_overlap"] / 6).mean() for k, _, _ in models]; hit_sd = [(rep[f"{k}_test_overlap"] / 6).std(ddof=1) for k, _, _ in models]
    ax.bar(xs - w / 2, score, w, yerr=score_sd, color=GREY, capsize=2, error_kw={"lw": 0.7}, label="mean model score")
    ax.bar(xs + w / 2, hit, w, yerr=hit_sd, color=GREEN, capsize=2, error_kw={"lw": 0.7}, label="realised hit rate")
    for x, s_, h_ in zip(xs, score, hit):
        ax.text(x - w / 2, s_ + 0.07, f"{s_:.3f}", ha="center", fontsize=7.5); ax.text(x + w / 2, h_ + 0.07, f"{h_:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(xs); ax.set_xticklabels([m[1] for m in models]); ax.set_ylim(0, 1.25); ax.set_ylabel("Selected numbers, test period")
    ax.set_title("Score versus realised hit rate"); ax.legend(loc="upper right", frameon=False, handlelength=1.2); panel(ax, "(c)")
    fig.tight_layout(w_pad=1.0)
    fig.savefig(Path(out) / "overfit_generalization.pdf", bbox_inches="tight"); fig.savefig(Path(out) / "overfit_generalization.png", bbox_inches="tight", dpi=300); plt.close(fig)


def fig_near_miss(data, out):
    ref = json.load(open(Path(data) / "C_ref.json")); runs = pd.read_csv(Path(data) / "C_runs.csv")
    fig = plt.figure(figsize=(FULLW, 2.85))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.12, 1], wspace=0.62)
    # (a) beta sweep
    ax = fig.add_subplot(gs[0]); sw = ref["quet_beta"]; ctrl = ref["ref_so_sanh"]["so_ky_tb_doi_chung"]
    b = np.array([r["beta"] for r in sw]); rel = np.array([r["tuong_doi_pct"] for r in sw])
    lo = np.array([r["ci95"][0] for r in sw]) / ctrl * 100; hi = np.array([r["ci95"][1] for r in sw]) / ctrl * 100
    ax.fill_between(b, lo, hi, color=ORANGE, alpha=0.25, lw=0, label="95% bootstrap")
    ax.plot(b, rel, color=ORANGE, marker="o", ms=3, lw=1.1, label="seed 42")
    rng = np.random.default_rng(2)
    ax.scatter(1 + rng.uniform(-0.07, 0.07, len(runs)), runs.rel_pct, s=7, color=BLUE, alpha=0.8, linewidths=0, zorder=4, label="Replicates, $\\beta=1$")
    ax.set_xlabel("$\\beta$"); ax.set_ylabel("Relative increase in rounds (%)"); ax.legend(loc="upper left", frameon=False, handlelength=1.3, fontsize=7.5); panel(ax, "(a)")
    # (b) heatmap
    ax = fig.add_subplot(gs[1]); g = ref["do_ben"]; M = np.array(g["tuong_doi_pct"])
    lab = {"tuyen_tinh": "Linear", "nhan_manh_gan_nhat": "Closest emph.", "tu_3_so_nhu_nhau": "Equal from 3", "chi_4_5_so": "Four and five"}
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(g["rho"]))); ax.set_xticklabels([f"{r:g}" for r in g["rho"]])
    ax.set_yticks(range(len(g["cach_gan_diem"]))); ax.set_yticklabels([lab[k] for k in g["cach_gan_diem"]])
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if M[i, j] < M.max() * 0.62 else "black")
    ax.set_xlabel("$\\rho$ (own control per column)"); ax.set_title("Relative increase (%), $\\beta=1$", fontsize=8.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03); cb.ax.tick_params(labelsize=7); panel(ax, "(b)")
    # (c) participation curves
    ax = fig.add_subplot(gs[2]); sc, ss = np.array(ref["survival_ctrl"]), np.array(ref["survival_sens"]); T = 70; t = np.arange(1, T + 1)
    ax.plot(t, 100 * sc[:T], color=BLUE, lw=1.1, label="control"); ax.plot(t, 100 * ss[:T], color=ORANGE, lw=1.1, label="sensitive ($\\beta=1$)")
    ax.fill_between(t, 100 * sc[:T], 100 * ss[:T], color=ORANGE, alpha=0.25, lw=0)
    ax.set_xlabel("Round $t$"); ax.set_ylabel("Still playing, $P(T\\geq t)$ (%)"); ax.legend(frameon=False, handlelength=1.3, loc="upper right"); panel(ax, "(c)")
    ins = ax.inset_axes([0.50, 0.30, 0.46, 0.33]); ins.plot(t, 100 * (ss[:T] - sc[:T]), color=ORANGE, lw=0.9)
    ins.axhline(0, color=GREY, lw=0.5); ins.set_title("Difference (pp)", fontsize=7, pad=1.5); ins.tick_params(labelsize=6.5, length=2, pad=1)
    for sp in ins.spines.values(): sp.set_linewidth(0.5)
    fig.savefig(Path(out) / "near_miss_sensitivity.pdf", bbox_inches="tight"); fig.savefig(Path(out) / "near_miss_sensitivity.png", bbox_inches="tight", dpi=300); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=EXPERIMENTS / "frozen_config.json")
    ap.add_argument("--data", type=Path, default=EXPERIMENTS / "paper_data")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/paper_figures")
    a = ap.parse_args(); cfg = json.loads(a.config.read_text(encoding="utf-8")); a.out.mkdir(parents=True, exist_ok=True)
    fig_concentration(cfg, a.data, a.out); fig_overfit(a.data, a.out); fig_near_miss(a.data, a.out)
    print("wrote", [p.name for p in Path(a.out).glob("*.pdf")])
