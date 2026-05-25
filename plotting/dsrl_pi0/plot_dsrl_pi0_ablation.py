"""Ablation figure for DSRL-pi0 TARC: bias x eval_mode x min_K cells + det-eval baselines.

Fetches:
  TARC ablation project (all 24 runs) -> groups by (sc, min_K, bias, eval_mode)
  Det-eval baseline project           -> apples-to-apples baseline under fully_deterministic noise
  Hybrid baseline project             -> DSRL default reference (QF=20 etc)

Produces:
  (a) Apples-to-apples Pareto: TARC bias_det_K1 vs baseline det QF=20  (the headline)
  (b) Bias-effect at min_K=1 det eval: bias=0 vs bias=2.0
  (c) Eval-mode-effect at min_K=5: hybrid (deteval project) vs fully_det (ablation v3)

Plus prints a tidy per-cell summary table.

Usage:
    python plot_dsrl_pi0_ablation.py
        [--tarc-ablation DSRL_pi0_Libero_TARC_ablation]
        [--baseline-det DSRL_pi0_Libero_baseline_deteval]
        [--baseline-hybrid arnavsukhija-eth-zurich/DSRL_pi0_Libero_baseline]
        [--tarc-hybrid DSRL_pi0_Libero_TARC_deteval]   # for the min_K=5 hybrid comparison
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

N_LAST_EVAL = 5
MIN_STEPS = 400_000


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

def _tail_mean(history, key):
    s = history[key].dropna()
    if s.empty:
        return float("nan")
    return float(s.tail(N_LAST_EVAL).mean())


def fetch_ablation_runs(project, verbose=True):
    """Pull TARC runs from the ablation project. Groups by (sc, min_K, bias, eval_mode)."""
    import wandb
    api = wandb.Api()
    runs = api.runs(project)

    # key -> list[per-seed dict]
    results = defaultdict(list)
    skipped = []

    for run in runs:
        c = run.config
        if not c.get("switch_cost_wrapper"):
            skipped.append((run.name, "not switchcost"))
            continue
        sc = c.get("switch_cost")
        min_K = c.get("min_time_between_switches")
        max_K = c.get("max_time_between_switches")
        bias = float(c.get("pseudo_time_init_bias", 0.0) or 0.0)
        eval_mode = c.get("eval_mode", "hybrid")
        seed = c.get("seed", -1)
        if sc is None or min_K is None:
            skipped.append((run.name, "missing sc/min_K"))
            continue

        history = run.history(
            keys=["evaluation/success_rate", "evaluation/avg_num_pi0_calls",
                  "evaluation/avg_K_per_decision", "evaluation/avg_episode_len",
                  "evaluation/avg_K_std_within_episode", "_step"],
            samples=2000,
        )
        if history.empty:
            skipped.append((run.name, "empty history"))
            continue
        eval_rows = history.dropna(subset=["evaluation/success_rate"])
        if eval_rows.empty:
            skipped.append((run.name, "no eval rows"))
            continue
        last_step = int(eval_rows["_step"].max()) if "_step" in eval_rows.columns else 0
        if last_step < MIN_STEPS:
            skipped.append((run.name, f"only {last_step} steps"))
            continue

        succ = _tail_mean(eval_rows, "evaluation/success_rate")
        calls = _tail_mean(eval_rows, "evaluation/avg_num_pi0_calls")
        avg_K = _tail_mean(eval_rows, "evaluation/avg_K_per_decision")
        K_std = _tail_mean(eval_rows, "evaluation/avg_K_std_within_episode") if "evaluation/avg_K_std_within_episode" in eval_rows.columns else float("nan")

        key = (float(sc), int(min_K), float(bias), str(eval_mode))
        results[key].append({
            "success": succ, "pi0_calls": calls, "avg_K": avg_K, "K_std": K_std,
            "seed": seed, "run_name": run.name, "last_step": last_step,
        })
        if verbose:
            print(f"  [keep] {run.name[:55]:55s}  sc={sc:>4g}  minK={min_K}  bias={bias}  "
                  f"eval={eval_mode:<20s} seed={seed}  succ={succ:.3f}  calls={calls:6.2f}  K={avg_K:5.2f}")

    if verbose and skipped:
        print(f"\n  skipped {len(skipped)} runs:")
        for n, r in skipped[:10]:
            print(f"    [skip] {n}: {r}")
    return dict(results)


def fetch_baseline_runs(project, eval_mode_filter=None, verbose=True):
    """Pull fixed-QF baseline runs, optionally filtered by eval_mode."""
    import wandb
    api = wandb.Api()
    runs = api.runs(project)
    results = defaultdict(list)
    skipped = []

    for run in runs:
        c = run.config
        if c.get("switch_cost_wrapper", 0):
            skipped.append((run.name, "switchcost on"))
            continue
        qf = c.get("query_freq")
        if qf is None or qf <= 0:
            skipped.append((run.name, "no QF"))
            continue
        run_eval_mode = c.get("eval_mode", "hybrid")
        if eval_mode_filter is not None and run_eval_mode != eval_mode_filter:
            skipped.append((run.name, f"eval_mode={run_eval_mode} != {eval_mode_filter}"))
            continue
        seed = c.get("seed", -1)

        history = run.history(
            keys=["evaluation/success_rate", "evaluation/avg_episode_len", "_step"],
            samples=2000,
        )
        if history.empty:
            skipped.append((run.name, "empty hist"))
            continue
        eval_rows = history.dropna(subset=["evaluation/success_rate"])
        if eval_rows.empty:
            skipped.append((run.name, "no eval rows"))
            continue
        last_step = int(eval_rows["_step"].max())
        if last_step < MIN_STEPS:
            skipped.append((run.name, f"only {last_step} steps"))
            continue

        succ = _tail_mean(eval_rows, "evaluation/success_rate")
        ep_len = _tail_mean(eval_rows, "evaluation/avg_episode_len")
        if np.isnan(succ) or np.isnan(ep_len):
            skipped.append((run.name, "nan"))
            continue
        calls = float(ep_len) / float(qf)
        if verbose:
            print(f"  [keep] {run.name[:55]:55s}  QF={qf:<3d} eval={run_eval_mode:<20s} seed={seed}  "
                  f"succ={succ:.3f}  ep_len={ep_len:5.1f}  calls={calls:6.2f}")

        results[int(qf)].append({
            "success": succ, "pi0_calls": calls, "episode_len": ep_len, "seed": seed,
            "run_name": run.name, "last_step": last_step,
        })

    if verbose and skipped:
        print(f"\n  skipped {len(skipped)} runs:")
        for n, r in skipped[:8]:
            print(f"    [skip] {n}: {r}")
    return dict(results)


def agg(seeds, keys=("success", "pi0_calls", "avg_K")):
    out = {"n": len(seeds)}
    for k in keys:
        vals = np.array([s[k] for s in seeds if s.get(k) is not None and not np.isnan(s.get(k, np.nan))], dtype=float)
        if vals.size == 0:
            out[k] = (np.nan, np.nan)
        else:
            m = float(vals.mean())
            sem = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
            out[k] = (m, sem)
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

CELL_STYLE = {
    # (min_K, bias, eval_mode): (label, color_idx, marker)
    (5, 0.0, "hybrid"):              ("min_K=5 baseline (hybrid)",   0, "o"),
    (5, 0.0, "fully_deterministic"): ("min_K=5 (det eval)",          0, "s"),
    (1, 0.0, "hybrid"):              ("min_K=1 no bias (hybrid)",    1, "o"),
    (1, 0.0, "fully_deterministic"): ("min_K=1 no bias (det)",       1, "s"),
    (1, 2.0, "hybrid"):              ("min_K=1 +bias (hybrid)",      3, "o"),
    (1, 2.0, "fully_deterministic"): ("min_K=1 +bias (det)",         3, "*"),
}


def make_figure(tarc_data, baseline_det, baseline_hybrid, tarc_hybrid_K5, output_dir):
    palette = sns.color_palette("colorblind")
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    # Plot each (min_K, bias, eval_mode) cell
    # Aggregate per sc value
    cells = defaultdict(list)  # key -> list of (sc, agg)
    for (sc, min_K, bias, eval_mode), seeds in tarc_data.items():
        cell_key = (min_K, bias, eval_mode)
        a = agg(seeds)
        cells[cell_key].append((sc, a))

    for cell_key, points in cells.items():
        if cell_key not in CELL_STYLE:
            print(f"  WARN: unknown cell {cell_key}, skipping plot")
            continue
        label, cidx, marker = CELL_STYLE[cell_key]
        color = palette[cidx]
        points.sort(key=lambda x: x[0])
        xs = [a["pi0_calls"][0] for _, a in points]
        xerr = [a["pi0_calls"][1] for _, a in points]
        ys = [a["success"][0] for _, a in points]
        yerr = [a["success"][1] for _, a in points]
        ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, marker=marker, ls="-", color=color,
                    label=label, markersize=7, capsize=2.0, lw=1.0)
        # annotate sc values
        for (sc, _), x, y in zip(points, xs, ys):
            ax.annotate(rf"$c{{=}}{sc:g}$", xy=(x, y), xytext=(4, 4),
                        textcoords="offset points", fontsize=6.5, color=color)

    # Optionally merge in tarc_hybrid_K5 (existing _deteval project, min_K=5 hybrid eval) for comparison
    if tarc_hybrid_K5:
        cell_key = (5, 0.0, "hybrid")
        if cell_key in CELL_STYLE:
            label, cidx, marker = CELL_STYLE[cell_key]
            color = palette[cidx]
            points = []
            for sc, seeds in tarc_hybrid_K5.items():
                a = agg(seeds)
                points.append((sc, a))
            points.sort(key=lambda x: x[0])
            xs = [a["pi0_calls"][0] for _, a in points]
            xerr = [a["pi0_calls"][1] for _, a in points]
            ys = [a["success"][0] for _, a in points]
            yerr = [a["success"][1] for _, a in points]
            ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, marker=marker, ls="--", color=color,
                        label=label, markersize=7, capsize=2.0, lw=1.0, alpha=0.6)

    # Baselines
    for project_data, suffix, ls, alpha, marker in [
        (baseline_det, "det", "-", 1.0, "D"),
        (baseline_hybrid, "hybrid", ":", 0.5, "x"),
    ]:
        if not project_data:
            continue
        qfs = sorted(project_data.keys())
        aggs = [agg(project_data[qf], keys=("success", "pi0_calls")) for qf in qfs]
        xs = [a["pi0_calls"][0] for a in aggs]
        ys = [a["success"][0] for a in aggs]
        xerr = [a["pi0_calls"][1] for a in aggs]
        yerr = [a["success"][1] for a in aggs]
        ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, marker=marker, ls=ls,
                    color=palette[2], label=f"baseline QF ({suffix})",
                    markersize=5, capsize=2.0, lw=1.0, alpha=alpha)
        for qf, x, y in zip(qfs, xs, ys):
            ax.annotate(f"QF={qf}", xy=(x, y), xytext=(3, -8),
                        textcoords="offset points", fontsize=6.0, color=palette[2], alpha=alpha)

    ax.set_xscale("log")
    ax.set_xlabel(r"Average $\pi_0$ inferences / episode")
    ax.set_ylabel("Success rate")
    ax.set_title(r"DSRL-$\pi_0$ on LIBERO: TARC ablation")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", fontsize=7, frameon=True, ncol=2)
    fig.tight_layout()

    pdf = os.path.join(output_dir, "DSRL_pi0_TARC_Ablation.pdf")
    png = pdf.replace(".pdf", ".png")
    plt.savefig(pdf)
    plt.savefig(png, dpi=300)
    print(f"--- saved: {pdf}")
    print(f"--- saved: {png}")


def print_summary(tarc_data, baseline_det, baseline_hybrid, tarc_hybrid_K5):
    print("\n=== TARC ablation (per cell) ===")
    rows = []
    for (sc, min_K, bias, eval_mode), seeds in sorted(tarc_data.items()):
        a = agg(seeds)
        rows.append((sc, min_K, bias, eval_mode, a["n"],
                     a["success"][0], a["success"][1],
                     a["pi0_calls"][0], a["pi0_calls"][1],
                     a["avg_K"][0]))
    print(f"{'sc':>4} {'minK':>5} {'bias':>5} {'eval':<22} {'n':>2}  "
          f"{'succ':>14}  {'pi0_calls':>14}  {'avg_K':>6}")
    for r in rows:
        sc, minK, bias, em, n, m1, s1, m2, s2, k = r
        print(f"{sc:>4g} {minK:>5} {bias:>5g} {em:<22} {n:>2}  "
              f"{m1:>6.3f}±{s1:6.3f}  {m2:>6.2f}±{s2:6.2f}  {k:>6.2f}")

    if tarc_hybrid_K5:
        print("\n=== TARC reference (existing deteval project, min_K=5 hybrid eval) ===")
        for sc, seeds in sorted(tarc_hybrid_K5.items()):
            a = agg(seeds)
            print(f"  c={sc:g}  n={a['n']}  succ={a['success'][0]:.3f}±{a['success'][1]:.3f}  "
                  f"calls={a['pi0_calls'][0]:.2f}±{a['pi0_calls'][1]:.2f}  K={a['avg_K'][0]:.2f}")

    if baseline_det:
        print("\n=== Baseline (eval=fully_deterministic) — apples-to-apples ===")
        for qf, seeds in sorted(baseline_det.items()):
            a = agg(seeds, keys=("success", "pi0_calls"))
            print(f"  QF={qf}  n={a['n']}  succ={a['success'][0]:.3f}±{a['success'][1]:.3f}  "
                  f"calls={a['pi0_calls'][0]:.2f}±{a['pi0_calls'][1]:.2f}")

    if baseline_hybrid:
        print("\n=== Baseline (eval=hybrid, DSRL default) ===")
        for qf, seeds in sorted(baseline_hybrid.items()):
            a = agg(seeds, keys=("success", "pi0_calls"))
            print(f"  QF={qf}  n={a['n']}  succ={a['success'][0]:.3f}±{a['success'][1]:.3f}  "
                  f"calls={a['pi0_calls'][0]:.2f}±{a['pi0_calls'][1]:.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tarc-ablation", default="DSRL_pi0_Libero_TARC_ablation")
    p.add_argument("--baseline-det", default="DSRL_pi0_Libero_baseline_deteval")
    p.add_argument("--baseline-hybrid", default="arnavsukhija-eth-zurich/DSRL_pi0_Libero_baseline")
    p.add_argument("--tarc-hybrid-K5", default="DSRL_pi0_Libero_TARC_deteval",
                   help="Existing deteval project; we use its min_K=5 runs as the hybrid-eval reference")
    p.add_argument("--output-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = p.parse_args()

    print(f"Fetching ablation TARC runs from {args.tarc_ablation}...")
    tarc_data = fetch_ablation_runs(args.tarc_ablation)

    print(f"\nFetching det baseline from {args.baseline_det}...")
    baseline_det = fetch_baseline_runs(args.baseline_det)

    print(f"\nFetching hybrid baseline from {args.baseline_hybrid}...")
    baseline_hybrid = fetch_baseline_runs(args.baseline_hybrid)

    # Reference: existing deteval project min_K=5 hybrid eval (the published v3 sanity reference)
    print(f"\nFetching min_K=5 hybrid reference from {args.tarc_hybrid_K5}...")
    tarc_ref = fetch_ablation_runs(args.tarc_hybrid_K5)
    # Filter to min_K=5, bias=0, eval_mode=hybrid
    tarc_hybrid_K5 = defaultdict(list)
    for (sc, minK, bias, em), seeds in tarc_ref.items():
        if minK == 5 and bias == 0.0 and em == "hybrid":
            tarc_hybrid_K5[sc].extend(seeds)

    print_summary(tarc_data, baseline_det, baseline_hybrid, dict(tarc_hybrid_K5))
    make_figure(tarc_data, baseline_det, baseline_hybrid, dict(tarc_hybrid_K5),
                output_dir=args.output_dir)


if __name__ == "__main__":
    main()
