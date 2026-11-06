"""Multi-task bar chart for DSRL-pi0 TARC results.

Shows TARC (sc=20, K∈[5,50]) vs fixed QF=20 baseline side-by-side for
multiple LIBERO-90 tasks in a single figure.

Panel (a): converged success rate per task
Panel (b): converged mean chunk-length K per task (efficiency)

Usage:
    python plot_dsrl_pi0_multitask_bar.py
    python plot_dsrl_pi0_multitask_bar.py --tasks 0 24 57
    python plot_dsrl_pi0_multitask_bar.py --from-cache
    python plot_dsrl_pi0_multitask_bar.py --dry-run    # print converged stats only
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    from tueplots import bundles, figsizes as tuefigsizes
    HAS_TUEPLOTS = True
except ImportError:
    HAS_TUEPLOTS = False

try:
    import scienceplots  # noqa: F401
    HAS_SCIENCEPLOTS = True
except ImportError:
    HAS_SCIENCEPLOTS = False

# Reuse data fetching / aggregation from the single-task matching plot.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_dsrl_pi0_matching import (
    fetch_tarc_curves,
    fetch_baseline_curves,
    extract_final_stats,
    _converged_kstats,
    N_LAST_FOR_CONVERGED,
    DEFAULT_TARC_PROJECT,
    DEFAULT_BASELINE_PROJECT,
)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Short human-readable labels for each task id.
TASK_LABELS = {
    0:  "Task 0",
    24: "Task 24",
    57: "Task 57",
    60: "Task 60",
    77: "Task 77",
    86: "Task 86",
}

SWITCH_COST = 20
BASELINE_QF = 20
MIN_K = 5
MAX_K = 50


def _apply_style():
    if HAS_TUEPLOTS:
        plt.rcParams.update(bundles.neurips2024())
    elif HAS_SCIENCEPLOTS:
        plt.style.use(["science", "ieee"])


def cache_path_for(task_id: int) -> str:
    return os.path.join(
        OUTPUT_DIR,
        f"DSRL_pi0_multitask_cache_task{task_id}_sc{SWITCH_COST:g}_qf{BASELINE_QF}.json",
    )


def load_task_data(task_id: int, from_cache: bool, verbose: bool = True) -> dict:
    """Fetch (or load) TARC + baseline curves for one task. Returns dict with keys
    'tarc' and 'baseline', each a list of per-seed dicts."""
    cpath = cache_path_for(task_id)
    if from_cache and os.path.exists(cpath):
        print(f"[task {task_id}] loading from cache: {cpath}")
        with open(cpath) as f:
            return json.load(f)

    print(f"\n[task {task_id}] fetching TARC  sc={SWITCH_COST} L=[{MIN_K},{MAX_K}] ...")
    tarc = fetch_tarc_curves(
        DEFAULT_TARC_PROJECT,
        switch_cost=SWITCH_COST,
        min_K=MIN_K,
        max_K=MAX_K,
        task_id=task_id,
        verbose=verbose,
    )
    print(f"[task {task_id}] fetching baseline QF={BASELINE_QF} ...")
    baseline = fetch_baseline_curves(
        DEFAULT_BASELINE_PROJECT,
        query_freq=BASELINE_QF,
        task_id=task_id,
        verbose=verbose,
    )
    payload = {"tarc": tarc, "baseline": baseline}
    with open(cpath, "w") as f:
        json.dump(payload, f)
    print(f"[task {task_id}] cached at: {cpath}")
    return payload


def converged_stats_for(task_data: dict) -> dict:
    """Return converged success rate and K stats for one task."""
    tarc_s = extract_final_stats(task_data["tarc"], ["success", "K_mean"])
    base_s = extract_final_stats(task_data["baseline"], ["success"])
    kstats = _converged_kstats(task_data["tarc"])
    K_mean = float(np.nanmean([s["K_mean"] for s in kstats])) if kstats else float("nan")
    K_sem  = (float(np.nanstd([s["K_mean"] for s in kstats], ddof=1) / np.sqrt(len(kstats)))
              if len(kstats) > 1 else 0.0)
    return {
        "tarc_sr":     tarc_s["success"][0],
        "tarc_sr_sem": tarc_s["success"][1],
        "base_sr":     base_s["success"][0],
        "base_sr_sem": base_s["success"][1],
        "K_mean":      K_mean,
        "K_sem":       K_sem,
    }


def make_multitask_figure(task_ids: list, stats_by_task: dict, output_dir: str):
    _apply_style()

    palette = sns.color_palette("colorblind")
    c_tarc = palette[0]
    c_base = palette[1]

    n = len(task_ids)
    x = np.arange(n)
    bw = 0.35          # width of each bar
    gap = 0.04         # gap between the two bars in a group
    ekw = {"lw": 0.8, "capsize": 2.5, "elinewidth": 0.8}
    bkw = {"width": bw, "edgecolor": "black", "linewidth": 0.5, "alpha": 0.85, "zorder": 3}

    if HAS_TUEPLOTS:
        plt.rcParams.update(tuefigsizes.neurips2024(nrows=1, ncols=2, height_to_width_ratio=0.85))
    fig, axes = plt.subplots(1, 2)

    # Panel (a): success rate
    ax = axes[0]
    tarc_srs = [stats_by_task[t]["tarc_sr"]     for t in task_ids]
    tarc_sem = [stats_by_task[t]["tarc_sr_sem"] for t in task_ids]
    base_srs = [stats_by_task[t]["base_sr"]     for t in task_ids]
    base_sem = [stats_by_task[t]["base_sr_sem"] for t in task_ids]

    ax.bar(x - bw/2 - gap/2, tarc_srs, color=c_tarc, yerr=tarc_sem, error_kw=ekw,
           label=rf"TARC ($c{{=}}{SWITCH_COST:g}$, $L{{\in}}[{MIN_K},{MAX_K}]$)", **bkw)
    ax.bar(x + bw/2 + gap/2, base_srs, color=c_base, yerr=base_sem, error_kw=ekw,
           label=rf"Fixed QF$={BASELINE_QF}$", **bkw)

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, f"Task {t}") for t in task_ids],
                       fontsize=6.5, rotation=15, ha="right")
    ax.set_ylabel(r"Success rate $\uparrow$", fontsize=8)
    ax.set_title("(a) Success rate across tasks", fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(labelsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="lower right", frameon=True, fontsize=6.0)

    # Panel (b): mean K (efficiency)
    ax = axes[1]
    K_means = [stats_by_task[t]["K_mean"] for t in task_ids]
    K_sems  = [stats_by_task[t]["K_sem"]  for t in task_ids]
    baseline_K = BASELINE_QF  # constant for all tasks

    ax.bar(x - bw/2 - gap/2, K_means, color=c_tarc, yerr=K_sems, error_kw=ekw,
           label="TARC (ours)", **bkw)
    # Baseline K is constant — draw as a flat bar aligned with TARC
    ax.bar(x + bw/2 + gap/2, [baseline_K] * n, color=c_base, **bkw,
           label="Fixed replanning interval")

    ax.axhline(MAX_K, color="0.55", lw=0.6, linestyle=":", label=rf"$L_{{\max}}={MAX_K}$")
    ax.axhline(MIN_K, color="0.55", lw=0.6, linestyle="-.", label=rf"$L_{{\min}}={MIN_K}$")

    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABELS.get(t, f"Task {t}") for t in task_ids],
                       fontsize=6.5, rotation=15, ha="right")
    ax.set_ylabel("Replanning interval $L$", fontsize=8)
    ax.set_title(r"(b) Adaptive replanning", fontsize=8)
    ax.set_ylim(0, MAX_K + 8)
    ax.tick_params(labelsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", frameon=True, fontsize=6.0)

    fig.tight_layout()

    task_str = "_".join(str(t) for t in task_ids)
    pdf_path = os.path.join(output_dir, f"DSRL_pi0_TARC_Multitask_Bar_tasks{task_str}.pdf")
    png_path = pdf_path.replace(".pdf", ".png")
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=300)
    print(f"\n--- saved: {pdf_path}")
    print(f"--- saved: {png_path}")
    return pdf_path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tasks", type=int, nargs="+", default=[0, 24, 57],
                        help="LIBERO-90 task IDs to include (default: 0 24 57)")
    parser.add_argument("--from-cache", action="store_true",
                        help="Load curves from local JSON cache instead of W&B")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print converged stats table without generating figures")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    stats_by_task = {}
    for tid in args.tasks:
        data = load_task_data(tid, from_cache=args.from_cache, verbose=True)
        stats_by_task[tid] = converged_stats_for(data)

    print("\n=== Converged stats ===")
    print(f"{'Task':>6}  {'TARC SR':>8}  {'±':>6}  {'Base SR':>8}  {'±':>6}  {'K_mean':>7}  {'±':>5}")
    for t in args.tasks:
        s = stats_by_task[t]
        print(f"{t:>6}  {s['tarc_sr']:>8.3f}  {s['tarc_sr_sem']:>6.3f}  "
              f"{s['base_sr']:>8.3f}  {s['base_sr_sem']:>6.3f}  "
              f"{s['K_mean']:>7.1f}  {s['K_sem']:>5.2f}")

    if not args.dry_run:
        make_multitask_figure(args.tasks, stats_by_task, args.output_dir)


if __name__ == "__main__":
    main()
