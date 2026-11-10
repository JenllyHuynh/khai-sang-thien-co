import argparse, itertools, json, math, sys, time
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

N = comb(55, 6)
RESULTS = []


def check(name, got, expected, tol):
    ok = abs(got - expected) <= tol
    RESULTS.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: recomputed {got:.6g} | paper {expected:.6g}")


# Experiment A
def analytic_A(cfg_a):
    w_cfg = cfg_a["weights"]
    lucky = set(cfg_a["lucky_numbers"])
    bmax = cfg_a["birthday_max"]
    w = np.array([w_cfg["lucky_weight"] if k in lucky else (w_cfg["birthday_weight"] if k <= bmax else w_cfg["base_weight"])
                  for k in range(1, 56)])
    W = w.sum()
    n_l = len(lucky)
    n_b = sum(1 for k in range(1, bmax + 1) if k not in lucky)
    n_o = 55 - n_l - n_b

    def p_combo(ws):  # successive sampling = Efraimidis-Spirakis keys
        tot = 0.0
        for perm in itertools.permutations(ws):
            rem, p = W, 1.0
            for x in perm:
                p *= x / rem
                rem -= x
            tot += p
        return tot

    types = []
    for l in range(0, 7):
        for b in range(0, 7 - l):
            o = 6 - l - b
            if l > n_l or b > n_b or o > n_o:
                continue
            cnt = comb(n_l, l) * comb(n_b, b) * comb(n_o, o)
            types.append((l, b, o, cnt, p_combo([w_cfg["lucky_weight"]] * l + [w_cfg["birthday_weight"]] * b + [w_cfg["base_weight"]] * o)))
    assert abs(sum(c * p for *_, c, p in types) - 1) < 1e-12
    out = {"total_weight": float(W), "n_lucky": n_l, "n_birthday_only": n_b, "n_other": n_o}
    for n in (5_000_000, 2_000_000):
        out[f"uniform_n{n}"] = {
            "share_pct": 100 * (1 - math.exp((n - 1) * math.log1p(-1 / N))),
            "unique_ratio": N * (1 - math.exp(n * math.log1p(-1 / N))) / n,
        }
        out[f"weighted_n{n}"] = {
            "share_pct": 100 * sum(c * p * (1 - math.exp((n - 1) * math.log1p(-p))) for *_, c, p in types),
            "unique_ratio": sum(c * (1 - math.exp(n * math.log1p(-p))) for *_, c, p in types) / n,
        }
    pmax = max(p for *_, p in types)
    out["n_compositions"] = len(types)
    top = sorted(types, key=lambda t: -t[4])[:6]  # six most probable composition classes
    rng = np.random.default_rng(1)
    per_run = np.array([max(rng.poisson(5_000_000 * p, size=c).max() for *_, c, p in top) for _ in range(2000)])
    out["poisson_max_cluster_per_run"] = {"mean": float(per_run.mean()), "sd": float(per_run.std()), "note": "six most probable composition classes, 2000 replicates"}
    out["max_combo_probability"] = pmax
    out["max_combo_ratio_to_uniform"] = pmax * N
    out["max_combo_expected_count_5e6"] = 5_000_000 * pmax
    return out


def run_A(cfg, outdir):
    a = cfg["A"]
    df = pd.read_csv(ROOT / a["csv"])
    print("\n== Experiment A ==")
    ok_cov = list(df.seed) == a["seeds"] and set(df.n_players_per_group) == {a["n_players_per_group"]}
    RESULTS.append(ok_cov); print(f"  [{'PASS' if ok_cov else 'FAIL'}] seeds 0..99 and 5e6 tickets/policy per row")
    s = {}
    for key, col in (("uniform", "vo_vi"), ("weighted", "tinh_cam")):
        c = df[f"{col}_pct_sharing"]
        s[key] = {"mean": c.mean(), "sd": c.std(ddof=1), "min": c.min(), "max": c.max(),
                  "unique_ratio_mean": df[f"{col}_unique_ratio"].mean(), "max_cluster_overall": int(df[f"{col}_max_cluster"].max()),
                  "max_cluster_mean_per_run": df[f"{col}_max_cluster"].mean()}
    s["ratio_of_means"] = s["weighted"]["mean"] / s["uniform"]["mean"]
    check("uniform mean %", s["uniform"]["mean"], 15.8408, 5e-5)
    check("uniform sd pp", s["uniform"]["sd"], 0.0192, 5e-5)
    check("uniform min %", s["uniform"]["min"], 15.7925, 5e-5); check("uniform max %", s["uniform"]["max"], 15.8850, 5e-5)
    check("weighted mean %", s["weighted"]["mean"], 46.0857, 5e-5)
    check("weighted sd pp", s["weighted"]["sd"], 0.0271, 5e-5)
    check("weighted min %", s["weighted"]["min"], 46.0195, 5e-5); check("weighted max %", s["weighted"]["max"], 46.1555, 5e-5)
    check("ratio of means", s["ratio_of_means"], 2.909, 5e-4)
    check("unique ratio uniform", s["uniform"]["unique_ratio_mean"], 0.918519, 5e-7)
    check("unique ratio weighted", s["weighted"]["unique_ratio_mean"], 0.709971, 5e-7)
    check("largest cluster uniform", s["uniform"]["max_cluster_overall"], 7, 0); check("largest cluster weighted", s["weighted"]["max_cluster_overall"], 45, 0)
    single = json.load(open(ROOT / a["single_example"]))
    check("2e6 uniform shared %", single["vo_vi"]["pct_players_sharing_ticket"], 6.6792, 5e-5)
    check("2e6 weighted shared %", single["tinh_cam"]["pct_players_sharing_ticket"], 27.4183, 5e-5)
    s["single_2e6"] = {"uniform": single["vo_vi"]["pct_players_sharing_ticket"], "weighted": single["tinh_cam"]["pct_players_sharing_ticket"]}
    s["analytic"] = analytic_A(a)
    an = s["analytic"]
    print("  analytic (not in the paper's claims table; reported in new text):")
    for k in ("uniform_n5000000", "weighted_n5000000", "uniform_n2000000", "weighted_n2000000"):
        print(f"    {k}: share {an[k]['share_pct']:.4f}%  unique ratio {an[k]['unique_ratio']:.6f}")
    print(f"    most likely combination: p={an['max_combo_probability']:.3e} = {an['max_combo_ratio_to_uniform']:.1f}x uniform, expected {an['max_combo_expected_count_5e6']:.1f} holders per 5e6 tickets")
    pm = an["poisson_max_cluster_per_run"]
    print(f"    compositions: {an['n_compositions']}; Poisson-model per-run max cluster {pm['mean']:.1f} (SD {pm['sd']:.1f}); observed mean per-run max cluster (weighted) {s['weighted']['max_cluster_mean_per_run']:.2f}")
    for k in ("uniform", "weighted"):
        n = 100; key = f"{k}_n5000000"
        s[f"se_{k}"] = s[k]["sd"] / np.sqrt(n); s[f"diff_{k}_pp"] = s[k]["mean"] - an[key]["share_pct"]
        print(f"    {k}: simulated mean - analytic = {s[f'diff_{k}_pp']:+.4f} pp (SE of mean {s[f'se_{k}']:.4f}); unique-ratio diff {s[k]['unique_ratio_mean']-an[key]['unique_ratio']:+.6f}")
    json.dump(s, open(outdir / "A_summary.json", "w"), indent=1, default=float)


# Experiment B
def run_B(cfg, outdir):
    from src.tran_phap_2_overfit.simulate import run_simulation
    from src.tran_phap_2_overfit.stats import summarize
    b = cfg["B"]
    print("\n== Experiment B ==")
    rows = []
    names = {"unres": "Lão Tặc AI (Học Vẹt)", "res": "Đạo Sĩ Khiêm Tốn (Baseline)"}
    for seed in b["seeds"] + [b["reference_seed"]]:
        sim = run_simulation(n_draws=b["n_draws"], window=b["window"], test_frac=b["test_frac"], seed=seed)
        s = summarize(sim)
        r = {"seed": seed, "n_train": s["n_train_draws"], "n_test": s["n_test_draws"]}
        a_ = sim["results"][names["unres"]]["test"].set_index("draw_index")["so_khop"]; b_ = sim["results"][names["res"]]["test"].set_index("draw_index")["so_khop"]
        r["test_corr_overlap"] = float(np.corrcoef(a_, b_)[0, 1]); r["test_same_overlap_frac"] = float((a_ == b_).mean())
        for tag, nm in names.items():
            for sp in ("train", "test"):
                m = s["models"][nm][sp]
                r[f"{tag}_{sp}_overlap"] = m["avg_matches"]; r[f"{tag}_{sp}_score"] = m["avg_confidence"]; r[f"{tag}_{sp}_gap"] = m["ao_tuong_gap"]
        rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(outdir / "B_runs.csv", index=False)
    rep = df[df.seed != b["reference_seed"]]
    ref = df[df.seed == b["reference_seed"]].iloc[0]
    check("train/test draws", rep.n_train.iloc[0] + rep.n_test.iloc[0], 1508, 0)
    check("n_train", rep.n_train.iloc[0], 1357, 0); check("n_test", rep.n_test.iloc[0], 151, 0)
    def ms(c): return rep[c].mean(), rep[c].std(ddof=1)
    for tag, nm in (("unres", "unrestricted"), ("res", "restricted")):
        pass
    check("unres train overlap", ms("unres_train_overlap")[0], 6.0, 1e-9)
    check("unres test overlap mean", ms("unres_test_overlap")[0], 0.6437, 5e-5); check("unres test overlap sd", ms("unres_test_overlap")[1], 0.0440, 5e-5)
    check("res train overlap mean", ms("res_train_overlap")[0], 0.6980, 5e-5); check("res train overlap sd", ms("res_train_overlap")[1], 0.0273, 5e-5)
    check("res test overlap mean", ms("res_test_overlap")[0], 0.6437, 5e-5); check("res test overlap sd", ms("res_test_overlap")[1], 0.0620, 5e-5)
    check("unres test score", ms("unres_test_score")[0], 0.99790, 5e-6); check("unres test score sd", ms("unres_test_score")[1], 0.01005, 5e-6)
    check("res test score", ms("res_test_score")[0], 0.11392, 5e-6); check("res test score sd", ms("res_test_score")[1], 0.00353, 5e-6)
    check("unres gap", ms("unres_test_gap")[0], 0.89062, 5e-6); check("unres gap sd", ms("unres_test_gap")[1], 0.01231, 5e-6)
    check("res gap", ms("res_test_gap")[0], 0.00663, 5e-6); check("res gap sd", ms("res_test_gap")[1], 0.00968, 5e-6)
    check("unres test range min", rep.unres_test_overlap.min(), 0.576159, 5e-7); check("unres test range max", rep.unres_test_overlap.max(), 0.754967, 5e-7)
    check("res test range min", rep.res_test_overlap.min(), 0.529801, 5e-7); check("res test range max", rep.res_test_overlap.max(), 0.761589, 5e-7)
    check("seed-42 unres test overlap", ref.unres_test_overlap, 0.635762, 5e-7); check("seed-42 res test overlap", ref.res_test_overlap, 0.523179, 5e-7)
    d = rep.unres_test_overlap - rep.res_test_overlap
    print(f"  extra: paired difference unres-res (test overlap): mean {d.mean():+.6f}, sd {d.std(ddof=1):.5f}; "
          f"sum of match counts {round((rep.unres_test_overlap*rep.n_test).sum())} vs {round((rep.res_test_overlap*rep.n_test).sum())}")
    print(f"  extra: mean within-history corr of per-draw overlaps {rep.test_corr_overlap.mean():.3f}; share of test draws with identical overlap {rep.test_same_overlap_frac.mean():.3f}")
    for tag in ("unres", "res"):
        se = rep[f"{tag}_test_overlap"].std(ddof=1) / np.sqrt(len(rep)); z = (36 / 55 - rep[f"{tag}_test_overlap"].mean()) / se
        print(f"  extra: {tag} test mean overlap SE {se:.5f}; (36/55 - mean)/SE = {z:.2f}")


# Experiment C
def run_C(cfg, outdir):
    from src.tran_phap_5_near_miss.config import ThamSo
    from src.tran_phap_5_near_miss.simulate import chay_thi_nghiem
    c = cfg["C"]; P = c["parameters"]
    ts = ThamSo(rho=P["rho"], alpha=P["alpha"], r_cap=P["r_cap"], cach_gan_diem=P["cach_gan_diem"], h0_mean=P["h0_mean"], h0_std=P["h0_std"],
                ngan_sach_median_ve=P["ngan_sach_median_ve"], ngan_sach_sigma=P["ngan_sach_sigma"], max_ky=P["max_ky"], gia_ve=P["gia_ve"],
                bang_thuong={int(k): v for k, v in c["payouts"].items()})
    print("\n== Experiment C ==")
    rows = []
    for seed in c["seeds"]:
        r = chay_thi_nghiem(c["n_agents"], ts, beta=c["beta"], seed=seed, n_boot=c["n_boot"]); s = r["so_sanh"]
        rows.append({"seed": seed, "ctrl": s["so_ky_tb_doi_chung"], "sens": s["so_ky_tb_nhay"], "extra": s["chenh_ky_tb"], "rel_pct": s["chenh_ky_tuong_doi_pct"],
                     "pct_longer": 100 * s["ty_le_agent_choi_lau_hon"], "pct_shorter": 100 * s["ty_le_agent_choi_it_hon"], "extra_spend": s["chenh_chi_tb"],
                     "pct_exposed": 100 * s["ty_le_agent_tung_gap_suyt_trung"]})
    df = pd.DataFrame(rows); df.to_csv(outdir / "C_runs.csv", index=False)
    def ms(col): return df[col].mean(), df[col].std(ddof=1)
    for col, label, m, sd in (("ctrl", "control rounds", 18.61584, 0.08559), ("sens", "sensitive rounds", 19.08506, 0.09256), ("extra", "extra rounds", 0.46922, 0.02721),
                              ("rel_pct", "relative increase %", 2.52052, 0.14517), ("pct_longer", "agents with extra rounds %", 4.739, 0.180), ("extra_spend", "extra expenditure", 4692.2, 272.1)):
        g = ms(col); tol = 5e-6 if col not in ("pct_longer", "extra_spend") else 5e-4
        check(f"{label} mean", g[0], m, tol if col != "extra_spend" else 0.05); check(f"{label} sd", g[1], sd, tol if col != "extra_spend" else 0.05)
    check("relative increase min", df.rel_pct.min(), 2.29308, 5e-6); check("relative increase max", df.rel_pct.max(), 2.75408, 5e-6)
    print(f"  extra: agents exposed to >=1 played round with S>0 (sensitive copy): {ms('pct_exposed')[0]:.2f}% (SD {ms('pct_exposed')[1]:.2f} pp across populations)")
    check("agents with fewer rounds (all seeds)", df.pct_shorter.abs().sum(), 0.0, 0.0)
    ref = chay_thi_nghiem(c["n_agents"], ts, beta=c["beta"], seed=c["reference_seed"], betas_quet=c["beta_sweep"], rhos_do_ben=c["rho_grid"], n_boot=c["n_boot"])
    s = ref["so_sanh"]
    check("seed-42 control rounds", s["so_ky_tb_doi_chung"], 18.7656, 5e-5); check("seed-42 sensitive rounds", s["so_ky_tb_nhay"], 19.2812, 5e-5)
    check("seed-42 extra rounds", s["chenh_ky_tb"], 0.5156, 5e-5); check("seed-42 relative %", s["chenh_ky_tuong_doi_pct"], 2.74758, 5e-6)
    check("seed-42 CI lo", s["chenh_ky_tb_ci95"][0], 0.46199, 5e-6); check("seed-42 CI hi", s["chenh_ky_tb_ci95"][1], 0.57594, 1e-5)
    check("beta=3 relative %", [r for r in ref["quet_beta"] if r["beta"] == 3.0][0]["tuong_doi_pct"], 8.42286, 5e-6)
    grid = ref["do_ben"]; i085 = grid["rho"].index(0.985)
    for exp, rule in zip((2.74758, 0.87874, 7.86865, 0.24246), grid["cach_gan_diem"]):
        check(f"grid rho=0.985 {rule}", grid["tuong_doi_pct"][grid["cach_gan_diem"].index(rule)][i085], exp, 5e-6)
    json.dump({"ref_so_sanh": s, "quet_beta": ref["quet_beta"], "do_ben": grid, "survival_ctrl": ref["survival"]["doi_chung"].tolist(),
               "survival_sens": ref["survival"]["nhay"].tolist(), "config": c}, open(outdir / "C_ref.json", "w"), default=float)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=EXPERIMENTS / "frozen_config.json")
    ap.add_argument("--out", type=Path, default=EXPERIMENTS / "paper_data")
    ap.add_argument("--skip", nargs="*", default=[], choices=["A", "B", "C"])
    args = ap.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8")); out = args.out; out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if "A" not in args.skip: run_A(cfg, out)
    if "B" not in args.skip: run_B(cfg, out)
    if "C" not in args.skip: run_C(cfg, out)
    print(f"\n{sum(RESULTS)}/{len(RESULTS)} checks passed in {time.time()-t0:.0f}s")
    sys.exit(0 if all(RESULTS) else 1)
