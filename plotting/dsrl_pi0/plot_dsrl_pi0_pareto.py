"""Pareto-style figure for the DSRL-pi0 TARC experiment.

Fetches two W&B projects:
  * TARC runs (`switch_cost_wrapper=1`)  ->  reads `evaluation/{success_rate, avg_num_pi0_calls, avg_K_per_decision}`
  * Baseline runs (fixed query_freq)     ->  reads `evaluation/{success_rate, avg_episode_len}`,
                                              derives pi0_calls per episode as `episode_len / query_freq`

Aggregates across seeds (mean / SEM from the last N_LAST_EVAL eval points to smooth out single-snapshot noise)
and produces a two-panel figure:
  (a) Pareto: success_rate vs avg_pi0_calls   for both methods on the same log-scale axes
  (b) avg_K vs switch_cost (TARC only)        with horizontal reference lines at each baseline's query_freq

Caches the raw aggregated numbers to `<output_dir>/DSRL_pi0_pareto_data.json` so the script can be re-run
without hitting W&B.

Usage:
    python plot_dsrl_pi0_pareto.py                                  # default projects, default style
    python plot_dsrl_pi0_pareto.py --tarc-project user/proj_TARC \\
                                   --baseline-project user/proj_BL  # custom W&B paths
    python plot_dsrl_pi0_pareto.py --from-cache                     # skip W&B, replot from JSON cache
"""
import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

try:
    import scienceplots  # noqa: F401
    HAS_SCIENCEPLOTS = True
except ImportError:
    HAS_SCIENCEPLOTS = False

DEFAULT_TARC_PROJECT = "arnavsukhija-eth-zurich/DSRL_pi0_Libero_TARC"
DEFAULT_BASELINE_PROJECT = "arnavsukhija-eth-zurich/DSRL_pi0_Libero_baseline"

# Average the last N eval points (instead of a single snapshot) to reduce single-eval noise
N_LAST_EVAL = 5

# Episodes are capped at this many env steps (used when episode_len isn't logged for some reason)
DEFAULT_EPISODE_CAP = 400


# ---------------------------------------------------------------------------
# W&B fetchers
# ---------------------------------------------------------------------------

def fetch_tarc_runs(project, max_K_filter=50, min_K_filter=1, min_steps=400_000, verbose=True):
    """Pull per-seed final-eval aggregates for TARC runs (switch_cost_wrapper=1).

    Filters:
      * switch_cost_wrapper == 1
      * max_time_between_switches == max_K_filter   (skip if None)
      * min_time_between_switches == min_K_filter   (skip if None)
      * run reached >= min_steps gradient steps (use last logged step as proxy)
    """
    import wandb
    api = wandb.Api()
    runs = api.runs(project)

    # results[switch_cost] = list of dicts (one per seed) with keys: success, pi0_calls, avg_K, seed
    results = defaultdict(list)
    skipped = []

    for run in runs:
        c = run.config
        if not c.get("switch_cost_wrapper"):
            skipped.append((run.name, "not switchcost"))
            continue
        run_max_K = c.get("max_time_between_switches")
        run_min_K = c.get("min_time_between_switches")
        if max_K_filter is not None and run_max_K != max_K_filter:
            skipped.append((run.name, f"max_K={run_max_K} != {max_K_filter}"))
            continue
        if min_K_filter is not None and run_min_K != min_K_filter:
            skipped.append((run.name, f"min_K={run_min_K} != {min_K_filter}"))
            continue

        sc = c.get("switch_cost")
        seed = c.get("seed", -1)
        if sc is None:
            skipped.append((run.name, "no switch_cost"))
            continue

        # Pull the last few eval points and average them
        history = run.history(
            keys=[
                "evaluation/success_rate",
                "evaluation/avg_num_pi0_calls",
                "evaluation/avg_K_per_decision",
                "_step",
            ],
            samples=2000,
        )
        if history.empty:
            skipped.append((run.name, "empty history"))
            continue

        # Drop rows missing the eval metrics
        eval_rows = history.dropna(subset=["evaluation/success_rate"])
        if eval_rows.empty:
            skipped.append((run.name, "no eval rows"))
            continue

        last_step = int(eval_rows["_step"].max()) if "_step" in eval_rows.columns else 0
        if min_steps is not None and last_step < min_steps:
            skipped.append((run.name, f"only {last_step} steps < {min_steps}"))
            continue

        tail = eval_rows.tail(N_LAST_EVAL)
        succ = tail["evaluation/success_rate"].dropna().mean()
        calls = tail["evaluation/avg_num_pi0_calls"].dropna().mean()
        avg_K = tail["evaluation/avg_K_per_decision"].dropna().mean()

        if np.isnan(succ) or np.isnan(calls):
            skipped.append((run.name, "nan in last eval"))
            continue

        if verbose:
            print(f"  [keep] {run.name:60s}  c={sc:<5g} seed={seed} max_K={run_max_K} "
                  f"last_step={last_step:>7d}  succ={succ:.3f}  calls={calls:6.2f}  K={avg_K:5.2f}")

        results[float(sc)].append({
            "success": float(succ),
            "pi0_calls": float(calls),
            "avg_K": float(avg_K) if not np.isnan(avg_K) else None,
            "seed": int(seed),
            "run_id": run.id,
            "run_name": run.name,
            "last_step": last_step,
        })

    if verbose and skipped:
        print(f"\n  skipped {len(skipped)} runs:")
        for name, reason in skipped[:30]:
            print(f"    [skip] {name:60s}  reason: {reason}")
        if len(skipped) > 30:
            print(f"    ... and {len(skipped) - 30} more")

    return dict(results)


def fetch_baseline_runs(project, min_steps=400_000, verbose=True):
    """Pull per-seed final-eval aggregates for fixed-query_freq baseline runs."""
    import wandb
    api = wandb.Api()
    runs = api.runs(project)

    results = defaultdict(list)
    skipped = []

    for run in runs:
        c = run.config
        if c.get("switch_cost_wrapper", 0):
            skipped.append((run.name, "switch_cost_wrapper on"))
            continue
        qf = c.get("query_freq")
        seed = c.get("seed", -1)
        if qf is None or qf <= 0:
            skipped.append((run.name, "no/zero query_freq"))
            continue

        history = run.history(
            keys=[
                "evaluation/success_rate",
                "evaluation/avg_episode_len",
                "_step",
            ],
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
        if min_steps is not None and last_step < min_steps:
            skipped.append((run.name, f"only {last_step} steps < {min_steps}"))
            continue

        tail = eval_rows.tail(N_LAST_EVAL)
        succ = tail["evaluation/success_rate"].dropna().mean()
        ep_len = tail["evaluation/avg_episode_len"].dropna().mean()

        if np.isnan(succ) or np.isnan(ep_len):
            skipped.append((run.name, "nan in last eval"))
            continue

        # pi0 calls per episode = (env steps per episode) / (env steps per pi0 query)
        # On a successful run episode_len < max_timesteps; this metric reflects the actual cost.
        derived_calls = float(ep_len) / float(qf)

        if verbose:
            print(f"  [keep] {run.name:60s}  QF={qf:<3d} seed={seed} "
                  f"last_step={last_step:>7d}  succ={succ:.3f}  ep_len={ep_len:5.1f}  calls={derived_calls:6.2f}")

        results[int(qf)].append({
            "success": float(succ),
            "pi0_calls": derived_calls,
            "episode_len": float(ep_len),
            "seed": int(seed),
            "run_id": run.id,
            "run_name": run.name,
            "last_step": last_step,
        })

    if verbose and skipped:
        print(f"\n  skipped {len(skipped)} runs:")
        for name, reason in skipped[:30]:
            print(f"    [skip] {name:60s}  reason: {reason}")
        if len(skipped) > 30:
            print(f"    ... and {len(skipped) - 30} more")

    return dict(results)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_seeds(per_seed_list, keys=("success", "pi0_calls")):
    """Mean / SEM across seeds for each named metric.

    Returns dict like {"success": (mean, sem), "pi0_calls": (mean, sem), "n": n_seeds}.
    """
    n = len(per_seed_list)
    out = {"n": n}
    for k in keys:
        vals = np.array([d[k] for d in per_seed_list if d.get(k) is not None], dtype=float)
        if vals.size == 0:
            out[k] = (np.nan, np.nan)
            continue
        m = float(vals.mean())
        s = float(vals.std(ddof=1) / np.sqrt(vals.size)) if vals.size > 1 else 0.0
        out[k] = (m, s)
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_figure(tarc_data, baseline_data, output_dir, dsrl_default_qf=20, annotate=True):
    """Build the (Pareto, K vs cost) two-panel figure and save PDF + PNG."""
    if HAS_SCIENCEPLOTS:
        plt.style.use(["science", "ieee"])

    palette = sns.color_palette("colorblind")
    color_tarc = palette[0]
    color_baseline = palette[1]
    color_marker = palette[3]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    # --- aggregate ---
    tarc_aggs = {sc: aggregate_seeds(v, keys=("success", "pi0_calls", "avg_K"))
                 for sc, v in sorted(tarc_data.items())}
    baseline_aggs = {qf: aggregate_seeds(v, keys=("success", "pi0_calls"))
                     for qf, v in sorted(baseline_data.items())}

    sc_keys = sorted(tarc_aggs.keys())
    qf_keys = sorted(baseline_aggs.keys())

    # --- Panel (a): Pareto scatter ---
    ax = axes[0]

    # baseline points
    bl_x = [baseline_aggs[qf]["pi0_calls"][0] for qf in qf_keys]
    bl_xerr = [baseline_aggs[qf]["pi0_calls"][1] for qf in qf_keys]
    bl_y = [baseline_aggs[qf]["success"][0] for qf in qf_keys]
    bl_yerr = [baseline_aggs[qf]["success"][1] for qf in qf_keys]
    # sort by x for a sensible connecting line
    order = np.argsort(bl_x)
    bl_x = [bl_x[i] for i in order]
    bl_xerr = [bl_xerr[i] for i in order]
    bl_y = [bl_y[i] for i in order]
    bl_yerr = [bl_yerr[i] for i in order]
    bl_qfs_sorted = [qf_keys[i] for i in order]

    ax.errorbar(bl_x, bl_y, xerr=bl_xerr, yerr=bl_yerr,
                marker="s", linestyle="--", color=color_baseline,
                label="Fixed query freq", markersize=4, capsize=1.5, lw=1.0)

    if annotate:
        for qf, x, y in zip(bl_qfs_sorted, bl_x, bl_y):
            label = rf"$\mathrm{{QF}}{{=}}{qf}$"
            ax.annotate(label, xy=(x, y), xytext=(3, -7),
                        textcoords="offset points", fontsize=5.5,
                        color=color_baseline)

    # TARC points
    tarc_x = [tarc_aggs[sc]["pi0_calls"][0] for sc in sc_keys]
    tarc_xerr = [tarc_aggs[sc]["pi0_calls"][1] for sc in sc_keys]
    tarc_y = [tarc_aggs[sc]["success"][0] for sc in sc_keys]
    tarc_yerr = [tarc_aggs[sc]["success"][1] for sc in sc_keys]
    order = np.argsort(tarc_x)
    tarc_x = [tarc_x[i] for i in order]
    tarc_xerr = [tarc_xerr[i] for i in order]
    tarc_y = [tarc_y[i] for i in order]
    tarc_yerr = [tarc_yerr[i] for i in order]
    sc_keys_sorted = [sc_keys[i] for i in order]

    ax.errorbar(tarc_x, tarc_y, xerr=tarc_xerr, yerr=tarc_yerr,
                marker="o", linestyle="-", color=color_tarc,
                label="TARC (adaptive)", markersize=4, capsize=1.5, lw=1.0)

    if annotate:
        for sc, x, y in zip(sc_keys_sorted, tarc_x, tarc_y):
            ax.annotate(rf"$c{{=}}{sc:g}$", xy=(x, y), xytext=(3, 4),
                        textcoords="offset points", fontsize=5.5,
                        color=color_tarc)

    # highlight the published DSRL operating point (QF=20)
    if dsrl_default_qf in baseline_aggs:
        x = baseline_aggs[dsrl_default_qf]["pi0_calls"][0]
        y = baseline_aggs[dsrl_default_qf]["success"][0]
        ax.scatter([x], [y], marker="*", s=80, color=color_marker, zorder=5,
                   edgecolor="black", linewidths=0.5,
                   label=f"DSRL default (QF={dsrl_default_qf})")

    ax.set_xscale("log")
    ax.set_xlabel(r"Average $\pi_0$ inferences / episode")
    ax.set_ylabel("Success rate")
    ax.set_title(r"(a) Pareto curve")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower left", frameon=True, fontsize=6.5)

    # --- Panel (b): avg_K vs switch_cost ---
    ax = axes[1]
    K_x = sc_keys  # in given order (small to large)
    K_y = [tarc_aggs[sc]["avg_K"][0] for sc in K_x]
    K_yerr = [tarc_aggs[sc]["avg_K"][1] for sc in K_x]

    ax.errorbar(K_x, K_y, yerr=K_yerr, marker="o", linestyle="-",
                color=color_tarc, markersize=4, capsize=1.5, lw=1.0)

    # reference horizontal lines at each baseline's QF — visual tie between adaptive K and fixed QF
    for qf in qf_keys:
        ax.axhline(y=qf, color=color_baseline, linestyle=":", alpha=0.5, lw=0.8)
        ax.text(min(K_x) * 0.55, qf, f"QF={qf}", fontsize=5.5,
                color=color_baseline, va="center", ha="right")

    ax.set_xscale("log")
    ax.set_xlabel(r"Switch cost $c$")
    ax.set_ylabel(r"Discovered avg chunk length $\bar K$")
    ax.set_title(r"(b) $\bar K(c)$")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()

    pdf_path = os.path.join(output_dir, "DSRL_pi0_TARC_Pareto.pdf")
    png_path = pdf_path.replace(".pdf", ".png")
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=300)
    print(f"--- saved: {pdf_path}")
    print(f"--- saved: {png_path}")
    return pdf_path


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def save_cache(tarc_data, baseline_data, cache_path):
    payload = {
        "tarc": {str(k): v for k, v in tarc_data.items()},
        "baseline": {str(k): v for k, v in baseline_data.items()},
    }
    with open(cache_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"--- cached W&B results at: {cache_path}")


def load_cache(cache_path):
    with open(cache_path) as f:
        payload = json.load(f)
    tarc = {float(k): v for k, v in payload["tarc"].items()}
    baseline = {int(k): v for k, v in payload["baseline"].items()}
    return tarc, baseline


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tarc-project", default=DEFAULT_TARC_PROJECT,
                        help="W&B project entity/name for TARC runs")
    parser.add_argument("--baseline-project", default=DEFAULT_BASELINE_PROJECT,
                        help="W&B project entity/name for baseline runs")
    parser.add_argument("--max-k", type=int, default=50,
                        help="Filter TARC runs to this max_time_between_switches value")
    parser.add_argument("--min-k", type=int, default=1,
                        help="Filter TARC runs to this min_time_between_switches value")
    parser.add_argument("--dsrl-default-qf", type=int, default=20,
                        help="QF value to highlight as the DSRL published default")
    parser.add_argument("--min-steps", type=int, default=400_000,
                        help="Only include runs that reached at least this many gradient steps "
                             "(filters out partial / debugging runs). Set 0 to disable.")
    parser.add_argument("--from-cache", action="store_true",
                        help="Skip W&B; load aggregated numbers from the JSON cache")
    parser.add_argument("--output-dir", default=os.path.dirname(os.path.abspath(__file__)),
                        help="Directory to save PDF/PNG/cache")
    args = parser.parse_args()

    cache_path = os.path.join(args.output_dir, "DSRL_pi0_pareto_data.json")

    if args.from_cache:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"No cache at {cache_path}; run without --from-cache first.")
        print(f"Loading cached data from {cache_path}")
        tarc_data, baseline_data = load_cache(cache_path)
    else:
        min_steps = args.min_steps if args.min_steps > 0 else None
        print(f"Fetching TARC runs from {args.tarc_project} "
              f"(max_K={args.max_k}, min_K={args.min_k}, min_steps={min_steps})...")
        tarc_data = fetch_tarc_runs(args.tarc_project,
                                    max_K_filter=args.max_k,
                                    min_K_filter=args.min_k,
                                    min_steps=min_steps)
        print(f"\nFetching baseline runs from {args.baseline_project} "
              f"(min_steps={min_steps})...")
        baseline_data = fetch_baseline_runs(args.baseline_project, min_steps=min_steps)
        save_cache(tarc_data, baseline_data, cache_path)

    # Summary print
    print("\n=== TARC (filtered) ===")
    for sc in sorted(tarc_data.keys()):
        seeds = tarc_data[sc]
        print(f"  c={sc:g}  n_seeds={len(seeds)}  "
              f"success={np.mean([s['success'] for s in seeds]):.3f}  "
              f"pi0_calls={np.mean([s['pi0_calls'] for s in seeds]):.2f}  "
              f"avg_K={np.mean([s['avg_K'] for s in seeds if s.get('avg_K') is not None]):.2f}")

    print("\n=== Baseline (fixed query_freq) ===")
    for qf in sorted(baseline_data.keys()):
        seeds = baseline_data[qf]
        print(f"  QF={qf}  n_seeds={len(seeds)}  "
              f"success={np.mean([s['success'] for s in seeds]):.3f}  "
              f"pi0_calls={np.mean([s['pi0_calls'] for s in seeds]):.2f}  "
              f"episode_len={np.mean([s['episode_len'] for s in seeds]):.1f}")

    make_figure(tarc_data, baseline_data,
                output_dir=args.output_dir,
                dsrl_default_qf=args.dsrl_default_qf)


if __name__ == "__main__":
    main()
