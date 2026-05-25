"""Diagnostic: per-step training curves for DSRL-pi0 TARC + baseline runs.

Answers the question: "is the last-5-eval mean hiding a peak earlier in training?"

Prints a per-run table of (peak / last / last-5) success and saves a 3-panel figure:
  (a) success_rate over training steps
  (b) evaluation/avg_num_pi0_calls over training steps
  (c) evaluation/avg_K_per_decision over training steps

Usage:
    python diagnose_dsrl_pi0_curves.py --tarc-project DSRL_pi0_Libero_TARC_deteval
"""
import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def fetch_history(project, switchcost_only):
    import wandb
    api = wandb.Api()
    runs = api.runs(project)
    out = []
    for run in runs:
        c = run.config
        is_sc = bool(c.get("switch_cost_wrapper"))
        if switchcost_only and not is_sc:
            continue
        if not switchcost_only and is_sc:
            continue
        keys = [
            "evaluation/success_rate",
            "evaluation/avg_num_pi0_calls",
            "evaluation/avg_K_per_decision",
            "evaluation/avg_episode_len",
            "_step",
        ]
        history = run.history(keys=keys, samples=5000)
        if history.empty:
            continue
        eval_rows = history.dropna(subset=["evaluation/success_rate"])
        if eval_rows.empty:
            continue
        out.append({
            "name": run.name,
            "sc": c.get("switch_cost"),
            "qf": c.get("query_freq"),
            "seed": c.get("seed", -1),
            "min_K": c.get("min_time_between_switches"),
            "max_K": c.get("max_time_between_switches"),
            "is_sc": is_sc,
            "steps": eval_rows["_step"].to_numpy(),
            "succ": eval_rows["evaluation/success_rate"].to_numpy(),
            "calls": eval_rows.get("evaluation/avg_num_pi0_calls", eval_rows["evaluation/success_rate"] * np.nan).to_numpy(),
            "K": eval_rows.get("evaluation/avg_K_per_decision", eval_rows["evaluation/success_rate"] * np.nan).to_numpy(),
            "ep_len": eval_rows.get("evaluation/avg_episode_len", eval_rows["evaluation/success_rate"] * np.nan).to_numpy(),
        })
    return out


def summarize(run):
    s = run["succ"]
    if s.size == 0:
        return None
    return {
        "peak_succ": float(np.max(s)),
        "peak_step": int(run["steps"][int(np.argmax(s))]),
        "last_succ": float(s[-1]),
        "last5_mean_succ": float(np.mean(s[-5:])) if s.size >= 1 else float(s[-1]),
        "n_evals": int(s.size),
        "last_step": int(run["steps"][-1]) if run["steps"].size else 0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tarc-project", default="DSRL_pi0_Libero_TARC_deteval")
    p.add_argument("--baseline-project", default="arnavsukhija-eth-zurich/DSRL_pi0_Libero_baseline")
    p.add_argument("--output-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = p.parse_args()

    print(f"Fetching TARC history from {args.tarc_project}...")
    tarc = fetch_history(args.tarc_project, switchcost_only=True)
    print(f"  -> {len(tarc)} switchcost runs")
    print(f"Fetching baseline history from {args.baseline_project}...")
    base = fetch_history(args.baseline_project, switchcost_only=False)
    print(f"  -> {len(base)} baseline runs")

    # --- Numerical summary ---
    print("\n=== TARC: peak vs last vs last-5 ===")
    print(f"{'run':70s} {'sc':>5s} {'minK':>5s} {'seed':>4s} {'peak':>6s}@{'step':>7s}  {'last':>6s}  {'last5':>6s}")
    for r in sorted(tarc, key=lambda x: (x.get("sc") or 0, x.get("min_K") or 0, x.get("seed") or 0)):
        s = summarize(r)
        if s is None:
            continue
        print(f"{r['name'][:70]:70s} {r.get('sc'):>5} {r.get('min_K'):>5} {r['seed']:>4} "
              f"{s['peak_succ']:>6.3f}@{s['peak_step']:>7d}  {s['last_succ']:>6.3f}  {s['last5_mean_succ']:>6.3f}")

    print("\n=== Baseline: peak vs last vs last-5 ===")
    print(f"{'run':70s} {'qf':>4s} {'seed':>4s} {'peak':>6s}@{'step':>7s}  {'last':>6s}  {'last5':>6s}")
    for r in sorted(base, key=lambda x: (x.get("qf") or 0, x.get("seed") or 0)):
        s = summarize(r)
        if s is None:
            continue
        print(f"{r['name'][:70]:70s} {r.get('qf'):>4} {r['seed']:>4} "
              f"{s['peak_succ']:>6.3f}@{s['peak_step']:>7d}  {s['last_succ']:>6.3f}  {s['last5_mean_succ']:>6.3f}")

    # --- Plot ---
    palette = sns.color_palette("tab10")
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

    # group TARC by (sc, min_K) for consistent colors
    tarc_groups = defaultdict(list)
    for r in tarc:
        key = (r.get("sc"), r.get("min_K"))
        tarc_groups[key].append(r)

    for i, (key, runs) in enumerate(sorted(tarc_groups.items(), key=lambda kv: (kv[0][0] or 0, kv[0][1] or 0))):
        color = palette[i % len(palette)]
        label = f"TARC c={key[0]:g} minK={key[1]}"
        for j, r in enumerate(runs):
            lbl = label if j == 0 else None
            axes[0].plot(r["steps"], r["succ"], color=color, alpha=0.6, lw=1.0, label=lbl)
            axes[1].plot(r["steps"], r["calls"], color=color, alpha=0.6, lw=1.0)
            axes[2].plot(r["steps"], r["K"], color=color, alpha=0.6, lw=1.0)

    # Baseline overlays (mean across seeds per qf, only show QF=20 as a reference)
    base_groups = defaultdict(list)
    for r in base:
        base_groups[r.get("qf")].append(r)
    for qf in [20, 50]:
        if qf not in base_groups:
            continue
        runs = base_groups[qf]
        color = "k" if qf == 20 else "gray"
        lbl = f"baseline QF={qf}"
        for j, r in enumerate(runs):
            axes[0].plot(r["steps"], r["succ"], color=color, alpha=0.35, lw=0.8, ls="--",
                         label=lbl if j == 0 else None)
            axes[1].plot(r["steps"], r["calls"], color=color, alpha=0.35, lw=0.8, ls="--")
            axes[2].plot(r["steps"], r["K"], color=color, alpha=0.35, lw=0.8, ls="--")

    axes[0].set_ylabel("success rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend(fontsize=7, loc="lower right", ncol=2)
    axes[0].grid(True, alpha=0.25)

    axes[1].set_ylabel(r"avg $\pi_0$ calls / episode")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.25)

    axes[2].set_ylabel(r"avg $K$ per decision")
    axes[2].grid(True, alpha=0.25)
    axes[2].set_xlabel("gradient step")

    fig.suptitle("DSRL-pi0 training curves (per-seed, raw)")
    fig.tight_layout()
    out_pdf = os.path.join(args.output_dir, "DSRL_pi0_training_curves.pdf")
    out_png = out_pdf.replace(".pdf", ".png")
    plt.savefig(out_pdf)
    plt.savefig(out_png, dpi=200)
    print(f"\n--- saved: {out_pdf}")
    print(f"--- saved: {out_png}")


if __name__ == "__main__":
    main()
