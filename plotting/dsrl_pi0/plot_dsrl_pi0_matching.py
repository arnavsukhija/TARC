"""Headline matching figure for the DSRL-pi0 TARC experiment.

Compares the *single best* TARC configuration against fixed-frequency baselines:

  Panel (a)  Success rate during training (mean ± SEM across seeds).
             TARC c=20, min_K=5, max_K=50   vs   fixed QF=20   vs   fixed QF=<matched>.

  Panel (b)  Adaptive chunk length K at convergence.
             Shows baseline bars at QF=20 (and optionally QF=matched) alongside the
             TARC within-episode K distribution (mean ± σ, [min, max] envelope).

The point: TARC matches (or exceeds) the published baseline's success rate with fewer
pi0 inferences per episode, and demonstrably varies K within each rollout.  The
matched-frequency baseline (QF ≈ mean TARC K) closes the "why not just use QF=32?"
reviewer objection.

Usage:
    python plot_dsrl_pi0_matching.py                             # task=57, sc=20, qf=20
    python plot_dsrl_pi0_matching.py --task-id 0                 # kitchen task
    python plot_dsrl_pi0_matching.py --task-id 86                # study task
    python plot_dsrl_pi0_matching.py --baseline-qf2 32           # add matched-freq baseline
    python plot_dsrl_pi0_matching.py --from-cache                # replot from JSON cache
"""
import argparse
import json
import os
from collections import defaultdict

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


def _apply_style():
    if HAS_TUEPLOTS:
        plt.rcParams.update(bundles.neurips2024())
    elif HAS_SCIENCEPLOTS:
        plt.style.use(["science", "ieee"])

DEFAULT_TARC_PROJECT = "arnavsukhija-eth-zurich/DSRL_pi0_Libero_TARC_ablation"
DEFAULT_BASELINE_PROJECT = "arnavsukhija-eth-zurich/DSRL_pi0_Libero_baseline_deteval"

POSTER_FS = 16   # font size (pt) of the poster variant

# Round _step to this many gradient steps when aligning seeds onto a common grid.
# eval_interval=10_000, so 10k is the natural bin width.
STEP_BIN = 10_000


# ---------------------------------------------------------------------------
# W&B fetchers — full training curves (not just last N evals)
# ---------------------------------------------------------------------------

def fetch_tarc_curves(project, switch_cost, min_K, max_K, task_id=57, verbose=True):
    """Pull per-seed full training curves for TARC runs matching (sc, min_K, max_K, task_id)."""
    import wandb
    api = wandb.Api(timeout=29)
    runs = api.runs(project)

    out = []
    skipped = []
    for run in runs:
        c = run.config
        if not c.get("switch_cost_wrapper"):
            continue
        if c.get("switch_cost") != switch_cost:
            continue
        if c.get("min_time_between_switches") != min_K:
            continue
        if c.get("max_time_between_switches") != max_K:
            continue
        # libero_task_id defaults to 57 in argparse, so old runs logged before the
        # multi-task extension will have it set to 57 even if not passed explicitly.
        run_task_id = int(c.get("libero_task_id", 57))
        if run_task_id != task_id:
            continue
        seed = c.get("seed", -1)

        # Core eval keys exist on every TARC run. Pull these first.
        core = run.history(
            keys=["evaluation/success_rate", "evaluation/avg_K_per_decision",
                  "evaluation/avg_episode_len", "_step"],
            samples=2000,
        )
        if core.empty or "evaluation/success_rate" not in core.columns:
            skipped.append((run.name, "no core eval column"))
            continue
        core = core.dropna(subset=["evaluation/success_rate"])
        if core.empty:
            skipped.append((run.name, "core history empty after dropna"))
            continue

        # K-trace stats (added later — older runs may lack these). Pull in a
        # separate call so a missing column doesn't blank out the whole query.
        kstats = run.history(
            keys=[
                "evaluation/avg_K_std_within_episode",
                "evaluation/avg_K_min_within_episode",
                "evaluation/avg_K_max_within_episode",
                "_step",
            ],
            samples=2000,
        )
        has_kstats = (not kstats.empty) and ("evaluation/avg_K_std_within_episode" in kstats.columns)

        steps = core["_step"].to_numpy().astype(int)

        def _core_col(name):
            if name not in core.columns:
                return [float("nan")] * len(steps)
            return core[name].to_numpy().astype(float).tolist()

        def _kcol(name):
            if not has_kstats or name not in kstats.columns:
                return [float("nan")] * len(steps)
            # Align kstats rows to core steps via dict lookup on _step.
            kmap = dict(zip(kstats["_step"].to_numpy().astype(int),
                            kstats[name].to_numpy().astype(float)))
            return [float(kmap.get(int(s), float("nan"))) for s in steps]

        out.append({
            "seed": int(seed),
            "run_name": run.name,
            "steps": steps.tolist(),
            "success": _core_col("evaluation/success_rate"),
            "K_mean": _core_col("evaluation/avg_K_per_decision"),
            "episode_len": _core_col("evaluation/avg_episode_len"),
            "K_std": _kcol("evaluation/avg_K_std_within_episode"),
            "K_min": _kcol("evaluation/avg_K_min_within_episode"),
            "K_max": _kcol("evaluation/avg_K_max_within_episode"),
            "has_kstats": has_kstats,
        })
        if verbose:
            ks = "kstats=yes" if has_kstats else "kstats=NO"
            print(f"  [keep] {run.name:60s} seed={seed} "
                  f"n_evals={len(steps)}  last_step={steps.max() if len(steps) else 0}  {ks}")

    if verbose and skipped:
        print(f"  skipped {len(skipped)} runs:")
        for name, reason in skipped[:30]:
            print(f"    [skip] {name:60s}  reason: {reason}")
    return out


def fetch_baseline_curves(project, query_freq, task_id=57, verbose=True):
    """Pull per-seed success-rate curves for fixed-QF baseline runs matching task_id."""
    import wandb
    api = wandb.Api(timeout=29)
    runs = api.runs(project)

    out = []
    skipped = []
    for run in runs:
        c = run.config
        if c.get("switch_cost_wrapper", 0):
            continue
        if c.get("query_freq") != query_freq:
            continue
        run_task_id = int(c.get("libero_task_id", 57))
        if run_task_id != task_id:
            continue
        seed = c.get("seed", -1)

        history = run.history(
            keys=["evaluation/success_rate", "evaluation/avg_episode_len", "_step"],
            samples=2000,
        )
        if history.empty or "evaluation/success_rate" not in history.columns:
            skipped.append((run.name, "no eval column"))
            continue
        h = history.dropna(subset=["evaluation/success_rate"])
        if h.empty:
            skipped.append((run.name, "empty history"))
            continue
        steps = h["_step"].to_numpy().astype(int)
        eplen = (h["evaluation/avg_episode_len"].to_numpy().astype(float).tolist()
                 if "evaluation/avg_episode_len" in h.columns
                 else [float("nan")] * len(steps))
        out.append({
            "seed": int(seed),
            "run_name": run.name,
            "steps": steps.tolist(),
            "success": h["evaluation/success_rate"].to_numpy().astype(float).tolist(),
            "episode_len": eplen,
        })
        if verbose:
            print(f"  [keep] {run.name:60s} seed={seed} "
                  f"n_evals={len(steps)}  last_step={steps.max() if len(steps) else 0}")

    if verbose and skipped:
        print(f"  skipped {len(skipped)} runs")
    return out


# ---------------------------------------------------------------------------
# Aggregation: align per-seed series onto a shared step grid, then mean ± SEM
# ---------------------------------------------------------------------------

def aggregate_curves(per_seed, fields, step_bin=STEP_BIN):
    """Bin each seed's series by `step_bin`, then aggregate across seeds.

    Returns dict {field: (steps, mean, sem)}.  `steps` is the sorted intersection
    of bins that *every* seed reached (so the curve never extrapolates).
    """
    if not per_seed:
        return {f: (np.array([]), np.array([]), np.array([])) for f in fields}

    # Step 1: for each seed, bin (step, value) pairs by rounded step.
    # Multiple eval points landing in the same bin get averaged (rarely happens
    # at 10k spacing with our eval_interval=10000 schedule, but it's safer).
    binned = []  # list of dicts: field -> {bin -> value}
    for seed_data in per_seed:
        steps = np.asarray(seed_data["steps"])
        rounded = (np.round(steps / step_bin) * step_bin).astype(int)
        d = {}
        for f in fields:
            vals = np.asarray(seed_data[f])
            mp = defaultdict(list)
            for r, v in zip(rounded, vals):
                if np.isfinite(v):
                    mp[int(r)].append(float(v))
            d[f] = {k: float(np.mean(vs)) for k, vs in mp.items()}
        binned.append(d)

    # Step 2 & 3: per-field, take bins where every seed has a value, then aggregate.
    # (Computed per-field so a sparsely-logged field doesn't kill the whole figure.)
    out = {}
    for f in fields:
        common = None
        for d in binned:
            seed_bins = set(d[f].keys())
            common = seed_bins if common is None else (common & seed_bins)
        common = sorted(common) if common else []
        if not common:
            out[f] = (np.array([]), np.array([]), np.array([]))
            continue
        stack = np.array([[d[f][b] for b in common] for d in binned])  # (n_seeds, n_bins)
        mean = stack.mean(axis=0)
        sem = stack.std(axis=0, ddof=1) / np.sqrt(stack.shape[0]) if stack.shape[0] > 1 else np.zeros_like(mean)
        out[f] = (np.array(common), mean, sem)
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

N_LAST_FOR_CONVERGED = 5      # # eval points to average for "converged" stats
EPISODE_TRIM_START = 200      # start panel (a) at this many training episodes
FALLBACK_EP_LEN = 250.0       # used only if avg_episode_len is missing


def extract_final_stats(per_seed, fields, n_last=N_LAST_FOR_CONVERGED):
    """Per-seed converged mean of `fields` → cross-seed (mean, SEM)."""
    per_seed_vals = {f: [] for f in fields}
    for d in per_seed:
        for f in fields:
            arr = np.array(d.get(f, []), dtype=float)
            finite = arr[np.isfinite(arr)]
            if finite.size:
                per_seed_vals[f].append(float(np.mean(finite[-n_last:])))
    result = {}
    for f, vals in per_seed_vals.items():
        if not vals:
            result[f] = (float("nan"), 0.0)
        elif len(vals) == 1:
            result[f] = (vals[0], 0.0)
        else:
            m = float(np.mean(vals))
            s = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
            result[f] = (m, s)
    return result


def compute_episode_mapping(per_seed, step_bin=STEP_BIN, fallback_eplen=FALLBACK_EP_LEN):
    """Build a {binned_step -> cumulative training episodes} dict for these runs.

    Cumulative episodes at step s ≈ Σ (Δstep / mean_episode_length_in_interval),
    averaged across seeds at each common bin.
    """
    seed_maps = []
    for d in per_seed:
        steps = np.asarray(d["steps"])
        eplens = np.asarray(d.get("episode_len", [fallback_eplen] * len(steps)), dtype=float)
        # Replace NaN / non-positive with fallback so the integral stays finite.
        eplens = np.where(np.isfinite(eplens) & (eplens > 0), eplens, fallback_eplen)
        rounded = (np.round(steps / step_bin) * step_bin).astype(int)

        bin_eplens = defaultdict(list)
        for r, el in zip(rounded, eplens):
            bin_eplens[int(r)].append(el)
        sorted_bins = sorted(bin_eplens.keys())
        avg_eplens = np.array([float(np.mean(bin_eplens[b])) for b in sorted_bins])
        # Episodes added in each bin ≈ step_bin / avg_eplen.  Cumsum gives cumulative.
        cum = np.cumsum(step_bin / avg_eplens)
        seed_maps.append({int(b): float(c) for b, c in zip(sorted_bins, cum)})

    if not seed_maps:
        return {}
    common = set(seed_maps[0].keys())
    for m in seed_maps[1:]:
        common &= set(m.keys())
    return {b: float(np.mean([m[b] for m in seed_maps])) for b in sorted(common)}


def _converged_kstats(tarc_curves, n_last=N_LAST_FOR_CONVERGED):
    """For each seed, average the last `n_last` non-NaN eval points of every K stat."""
    out = []
    for d in tarc_curves:
        def _last(name):
            arr = np.asarray(d[name], dtype=float)
            arr = arr[np.isfinite(arr)]
            return float(np.mean(arr[-n_last:])) if arr.size else float("nan")
        out.append({
            "seed": d["seed"],
            "K_mean": _last("K_mean"),
            "K_std":  _last("K_std"),
            "K_min":  _last("K_min"),
            "K_max":  _last("K_max"),
        })
    return out


def make_figure(tarc_curves, baseline_curves, output_dir, tarc_sc, baseline_qf,
                tarc_min_K, tarc_max_K, task_id=57,
                baseline2_curves=None, baseline2_qf=None, baseline2_final_val=None):
    _apply_style()

    palette = sns.color_palette("colorblind")
    c_tarc  = palette[0]
    c_base  = palette[1]
    c_base2 = palette[2]  # matched-frequency baseline

    if HAS_TUEPLOTS:
        plt.rcParams.update(tuefigsizes.neurips2024(nrows=1, ncols=2, height_to_width_ratio=0.87))
    fig, axes = plt.subplots(2, 1)

    # --- aggregate for panel (a) ---
    tarc_agg  = aggregate_curves(tarc_curves,     fields=("success",))
    base_agg  = aggregate_curves(baseline_curves, fields=("success",))
    tarc_ep   = compute_episode_mapping(tarc_curves)
    base_ep   = compute_episode_mapping(baseline_curves)
    if baseline2_curves:
        base2_agg = aggregate_curves(baseline2_curves, fields=("success",))
        base2_ep  = compute_episode_mapping(baseline2_curves)
    else:
        base2_agg = {"success": (np.array([]), np.array([]), np.array([]))}
        base2_ep  = {}

    def steps_to_episodes(steps_arr, ep_map):
        out_x = []
        for si in steps_arr:
            ep = ep_map.get(int(si))
            out_x.append(ep if ep is not None else float(si) / FALLBACK_EP_LEN)
        return np.asarray(out_x)

    # --- Panel (a): Success rate vs training episodes ---
    ax = axes[0]
    max_x_per_curve = []
    if tarc_agg["success"][0].size:
        s, m, e = tarc_agg["success"]
        x = steps_to_episodes(s, tarc_ep)
        ax.plot(x, m, color=c_tarc, lw=1.4,
                label=rf"TARC ($c{{=}}{tarc_sc:g}$, $L{{\in}}[{tarc_min_K},{tarc_max_K}]$)")
        ax.fill_between(x, m - e, m + e, color=c_tarc, alpha=0.22, linewidth=0)
        max_x_per_curve.append(float(x.max()))
    if base_agg["success"][0].size:
        s, m, e = base_agg["success"]
        x = steps_to_episodes(s, base_ep)
        ax.plot(x, m, color=c_base, lw=1.4, linestyle="--",
                label=rf"Fixed QF$={baseline_qf}$ (published)")
        ax.fill_between(x, m - e, m + e, color=c_base, alpha=0.22, linewidth=0)
        max_x_per_curve.append(float(x.max()))
    if baseline2_curves and base2_agg["success"][1].size:
        _, m2, _ = base2_agg["success"]
        if baseline2_final_val is not None:
            conv_val = baseline2_final_val
        else:
            conv_val = float(np.mean(m2[-N_LAST_FOR_CONVERGED:]))
        ax.axhline(conv_val, color=c_base2, lw=1.2, linestyle="-.",
                   label=rf"Fixed QF$={baseline2_qf}$ (matched $\bar{{L}}$) $\to$ {conv_val:.2f}")

    # Truncate to the shorter curve so both end at the same x.
    upper = min(max_x_per_curve) if max_x_per_curve else None
    ax.set_xlim(EPISODE_TRIM_START, upper)
    ax.set_xlabel("Training episodes", fontsize=8)
    ax.set_ylabel("Success rate", fontsize=8)
    ax.set_title(f"(a) Success rate — task {task_id}", fontsize=8)
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=True, fontsize=6.5)

    # --- Panel (b): Converged within-episode K ---
    ax = axes[1]
    kstats = _converged_kstats(tarc_curves)
    have_kstats = any(np.isfinite(s["K_std"]) for s in kstats)

    if have_kstats:
        K_mean = float(np.nanmean([s["K_mean"] for s in kstats]))
        K_std  = float(np.nanmean([s["K_std"]  for s in kstats]))
        K_min  = float(np.nanmean([s["K_min"]  for s in kstats]))
        K_max  = float(np.nanmean([s["K_max"]  for s in kstats]))
        n_seeds = sum(np.isfinite(s["K_mean"]) for s in kstats)
        K_mean_sem = (float(np.nanstd([s["K_mean"] for s in kstats], ddof=1)) / np.sqrt(n_seeds)
                      if n_seeds > 1 else 0.0)

        half_w = 0.32
        # Column positions depend on whether a second baseline is present.
        has_b2 = (baseline2_curves is not None) and (baseline2_qf is not None)
        if has_b2:
            x_base, x_base2, x_tarc = 0, 1, 2
        else:
            x_base, x_tarc = 0, 1

        # --- Baseline column(s): thin filled bar at constant K = QF ---
        ax.bar([x_base], [0.6], bottom=[baseline_qf - 0.3], width=2 * half_w,
               color=c_base, alpha=0.85, edgecolor="black", linewidth=0.5, zorder=3)
        ax.text(x_base, baseline_qf - 1.5,
                rf"$L{{\equiv}}{baseline_qf}$",
                ha="center", va="top", fontsize=6.0, color=c_base)
        if has_b2:
            ax.bar([x_base2], [0.6], bottom=[baseline2_qf - 0.3], width=2 * half_w,
                   color=c_base2, alpha=0.85, edgecolor="black", linewidth=0.5, zorder=3)
            ax.text(x_base2, baseline2_qf - 1.5,
                    rf"$L{{\equiv}}{baseline2_qf}$",
                    ha="center", va="top", fontsize=6.0, color=c_base2)

        # --- TARC column: outer band [K_min, K_max], inner band K_mean ± σ ---
        ax.bar([x_tarc], [K_max - K_min], bottom=[K_min], width=2 * half_w,
               color=c_tarc, alpha=0.22, edgecolor="none", zorder=2)
        ax.bar([x_tarc], [2 * K_std], bottom=[K_mean - K_std], width=2 * half_w,
               color=c_tarc, alpha=0.55, edgecolor="none", zorder=3)
        ax.plot([x_tarc - half_w, x_tarc + half_w], [K_mean, K_mean],
                color=c_tarc, lw=2.2, zorder=4)
        ax.errorbar([x_tarc], [K_mean], yerr=K_mean_sem,
                    fmt="o", color="black", ecolor="black",
                    markersize=3.5, capsize=2.5, lw=0.8, zorder=5)

        # Policy bound reference lines
        ax.axhline(tarc_max_K, color="0.55", lw=0.5, linestyle=":")
        ax.axhline(tarc_min_K, color="0.55", lw=0.5, linestyle=":")
        ax.text(x_tarc + half_w + 0.05, tarc_max_K,
                rf"$L_{{\max}}{{=}}{tarc_max_K}$",
                fontsize=5.5, color="0.4", va="center", ha="left")
        ax.text(x_tarc + half_w + 0.05, tarc_min_K,
                rf"$L_{{\min}}{{=}}{tarc_min_K}$",
                fontsize=5.5, color="0.4", va="center", ha="left")
        ax.text(x_tarc, max(K_min - 3, 1.5),
                rf"$\bar L{{=}}{K_mean:.1f}\,\pm\,{K_std:.1f}$"
                "\n"
                rf"$[{K_min:.0f},\,{K_max:.0f}]$",
                ha="center", va="top", fontsize=6.0, color=c_tarc)

        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=c_tarc, alpha=0.55, label=r"$\bar K \pm \sigma_K$"),
            Patch(facecolor=c_tarc, alpha=0.22, label=r"$[K_{\min}, K_{\max}]$"),
            Line2D([0], [0], color=c_tarc, lw=2.2, label=r"$\bar K$ (mean over seeds)"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", frameon=True, fontsize=6.0)

        if has_b2:
            ax.set_xticks([x_base, x_base2, x_tarc])
            ax.set_xticklabels([f"Baseline\nQF={baseline_qf}",
                                 f"Matched\nQF={baseline2_qf}",
                                 rf"TARC ($c{{=}}{tarc_sc:g}$)"])
            ax.set_xlim(-0.7, 2.95)
        else:
            ax.set_xticks([x_base, x_tarc])
            ax.set_xticklabels([f"Baseline\nQF={baseline_qf}",
                                 rf"TARC ($c{{=}}{tarc_sc:g}$)"])
            ax.set_xlim(-0.7, 1.95)
        ax.set_ylim(0, tarc_max_K + 5)
    else:
        ax.text(0.5, 0.5, "K-stats not logged for these runs",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=8, color="0.4")

    ax.set_ylabel(r"Chunk length $L$ at convergence", fontsize=8)
    ax.set_title("(b) Constant $L$ vs. adaptive $L$ within episode", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, axis="y", alpha=0.25)

    qf2_tag = f"_qf2-{baseline2_qf}" if baseline2_qf is not None else ""
    pdf_path = os.path.join(output_dir, f"DSRL_pi0_TARC_Matching_task{task_id}{qf2_tag}.pdf")
    png_path = pdf_path.replace(".pdf", ".png")
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=300)
    print(f"--- saved: {pdf_path}")
    print(f"--- saved: {png_path}")
    return pdf_path


# ---------------------------------------------------------------------------
# Bar figure: penalized SR | unpenalized SR | K-distribution (adaptivity)
# ---------------------------------------------------------------------------

def make_bar_figure(tarc_curves, baseline_curves, output_dir, tarc_sc, baseline_qf,
                    tarc_min_K, tarc_max_K, task_id=57,
                    baseline2_curves=None, baseline2_qf=None, baseline2_final_val=None,
                    poster=False):
    """2-panel bar chart: success rate | adaptive K distribution.

    poster=True: 8.0 x 3.2 in figure, 24 pt text, TARC orange / fixed interval grey,
    no panel titles; written to *_poster.{pdf,png}.
    """
    _apply_style()
    # 24 pt does not fit in an 8.0 x 3.2 in two-panel figure; 16 pt is the largest that does.
    F_LAB, F_TICK, F_ANN = (POSTER_FS, POSTER_FS, POSTER_FS) if poster else (8, 7, 6.5)
    F_XT = POSTER_FS if poster else 6.0

    palette = sns.color_palette("colorblind")
    c_tarc  = palette[1]   # orange, matches the other plots
    c_base  = "0.55" if poster else palette[0]   # grey on the poster, blue otherwise
    c_base2 = palette[2]

    # --- Converged stats from cache ---
    tarc_s = extract_final_stats(tarc_curves,     ["success", "K_mean", "episode_len"])
    base_s = extract_final_stats(baseline_curves, ["success", "episode_len"])
    has_b2 = (baseline2_curves is not None) and (baseline2_qf is not None)
    base2_s = extract_final_stats(baseline2_curves, ["success", "episode_len"]) if has_b2 else {}

    tarc_sr, tarc_sr_sem = tarc_s["success"]
    base_sr, base_sr_sem = base_s["success"]

    labels  = ["TARC (ours)", "Fixed replanning interval"]
    colors  = [c_tarc, c_base]
    sr_m    = [tarc_sr,  base_sr]
    sr_s    = [tarc_sr_sem, base_sr_sem]

    if has_b2:
        b2_sr     = baseline2_final_val if baseline2_final_val is not None else base2_s["success"][0]
        b2_sr_sem = 0.0 if baseline2_final_val is not None else base2_s["success"][1]
        labels.append("Fixed replanning interval");  colors.append(c_base2)
        sr_m.append(b2_sr);   sr_s.append(b2_sr_sem)

    x   = np.arange(len(labels))
    bw  = 0.55
    ekw = {"lw": 0.8, "capsize": 2.5, "elinewidth": 0.8}
    bkw = {"width": bw, "edgecolor": "black", "linewidth": 0.5, "alpha": 0.85, "zorder": 3}

    if poster:
        plt.rcParams.update({'text.usetex': False, 'font.family': 'sans-serif',
                             'font.sans-serif': ['DejaVu Sans'], 'mathtext.fontset': 'dejavusans'})
        fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
    else:
        if HAS_TUEPLOTS:
            plt.rcParams.update(tuefigsizes.neurips2024(nrows=1, ncols=2, height_to_width_ratio=0.9))
        fig, axes = plt.subplots(1, 2)

    def _bar(ax, means, sems, ylabel, title, ylim=None):
        for i, (m, s, c) in enumerate(zip(means, sems, colors)):
            ax.bar(i, m, color=c, yerr=s, error_kw=ekw, **bkw)
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace("Fixed replanning interval", "Fixed\ninterval" if poster else "Fixed replanning\ninterval") for l in labels],
                           fontsize=F_XT, ha="center")
        ax.set_ylabel(ylabel, fontsize=F_LAB)
        if not poster:
            ax.set_title(title, fontsize=F_LAB)
        if ylim:
            ax.set_ylim(*ylim)
        if poster:
            ax.set_yticks([0.0, 0.5, 1.0])
        ax.tick_params(labelsize=F_TICK)
        ax.grid(True, axis="y", alpha=0.25)

    _bar(axes[0], sr_m, sr_s,
         r"Success rate $\uparrow$",
         "(a) Success rate",
         ylim=(0.0, 1.05))

    # --- Panel (b): K-distribution (adaptivity) ---
    ax = axes[1]
    kstats    = _converged_kstats(tarc_curves)
    have_kstats = any(np.isfinite(s["K_std"]) for s in kstats)
    if have_kstats:
        K_mean_ = float(np.nanmean([s["K_mean"] for s in kstats]))
        K_std_  = float(np.nanmean([s["K_std"]  for s in kstats]))
        K_min_  = float(np.nanmean([s["K_min"]  for s in kstats]))
        K_max_  = float(np.nanmean([s["K_max"]  for s in kstats]))
        n_s     = sum(np.isfinite(s["K_mean"]) for s in kstats)
        K_mean_sem = (float(np.nanstd([s["K_mean"] for s in kstats], ddof=1)) / np.sqrt(n_s)
                      if n_s > 1 else 0.0)
        half_w = 0.32
        x_tarc, x_base = 0, 1                      # same order as panel (a)
        x_base2 = 2 if has_b2 else None
        x_last  = x_base2 if has_b2 else x_base

        ax.bar([x_base], [0.6], bottom=[baseline_qf - 0.3], width=2*half_w,
               color=c_base, alpha=0.85, edgecolor="black", linewidth=0.5, zorder=3)
        ax.text(x_base, baseline_qf - 1.5, f"{baseline_qf}",
                ha="center", va="top", fontsize=F_ANN, color=c_base)
        if has_b2 and x_base2 is not None:
            ax.bar([x_base2], [0.6], bottom=[baseline2_qf - 0.3], width=2*half_w,
                   color=c_base2, alpha=0.85, edgecolor="black", linewidth=0.5, zorder=3)
            ax.text(x_base2, baseline2_qf - 1.5, f"{baseline2_qf}",
                    ha="center", va="top", fontsize=F_ANN, color=c_base2)
        ax.bar([x_tarc], [K_max_ - K_min_], bottom=[K_min_], width=2*half_w,
               color=c_tarc, alpha=0.22, edgecolor="none", zorder=2)
        ax.plot([x_tarc - half_w, x_tarc + half_w], [K_mean_, K_mean_],
                color=c_tarc, lw=2.2, zorder=4)
        ax.text(x_tarc, max(K_min_ - 3, 1.5),
                (f"avg. {K_mean_:.0f}" if poster
                 else f"avg. {K_mean_:.0f}\n(range {K_min_:.0f}\u2013{K_max_:.0f})"),
                ha="center", va="top", fontsize=F_ANN, color=c_tarc)

        fx = "Fixed\ninterval" if poster else "Fixed replanning\ninterval"
        tick_labels = ["TARC (ours)", fx] + ([fx] if has_b2 else [])
        ax.set_xticks([x_tarc, x_base] + ([x_base2] if has_b2 else []))
        ax.set_xticklabels(tick_labels, fontsize=F_XT, ha="center")
        ax.set_xlim(-0.7, x_last + 0.7)
        ax.set_ylim(0, max(tarc_max_K, K_max_) + 5)

    ax.set_ylabel("Replanning interval\n(steps)" if poster else "Replanning interval (steps)",
                  fontsize=F_LAB)
    if not poster:
        ax.set_title("(b) Replanning interval", fontsize=F_LAB)
    ax.tick_params(labelsize=F_TICK)
    ax.grid(True, axis="y", alpha=0.25)

    qf2_tag  = f"_qf2-{baseline2_qf}" if has_b2 else ""
    poster_tag = "_poster" if poster else ""
    pdf_path = os.path.join(output_dir, f"DSRL_pi0_TARC_Bar_task{task_id}{qf2_tag}{poster_tag}.pdf")
    png_path = pdf_path.replace(".pdf", ".png")
    if poster:
        fig.tight_layout(pad=0.3, w_pad=1.5)
    plt.savefig(pdf_path)
    plt.savefig(png_path, dpi=300)
    print(f"--- saved: {pdf_path}")
    print(f"--- saved: {png_path}")

    # Also export each panel as its own file (same styling, cropped to the panel + labels).
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax_i, name in zip(axes, ["a_success", "b_replanning"]):
        bbox = ax_i.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        bbox = bbox.padded(0.03)
        base = pdf_path.replace(".pdf", f"_{name}")
        fig.savefig(base + ".pdf", bbox_inches=bbox)
        fig.savefig(base + ".png", dpi=300, bbox_inches=bbox)
        print(f"--- saved: {base}.png")
    return pdf_path


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------

def save_cache(tarc_curves, baseline_curves, cache_path, baseline2_curves=None):
    payload = {"tarc": tarc_curves, "baseline": baseline_curves}
    if baseline2_curves is not None:
        payload["baseline2"] = baseline2_curves
    with open(cache_path, "w") as f:
        json.dump(payload, f)
    print(f"--- cached W&B curves at: {cache_path}")


def load_cache(cache_path):
    with open(cache_path) as f:
        payload = json.load(f)
    return payload["tarc"], payload["baseline"], payload.get("baseline2")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tarc-project", default=DEFAULT_TARC_PROJECT)
    parser.add_argument("--baseline-project", default=DEFAULT_BASELINE_PROJECT)
    parser.add_argument("--task-id", type=int, default=57,
                        help="LIBERO-90 task id to plot (default: 57, the cream-cheese task)")
    parser.add_argument("--tarc-sc", type=float, default=20.0,
                        help="Switch cost of the TARC run to plot (default: 20)")
    parser.add_argument("--tarc-min-k", type=int, default=5)
    parser.add_argument("--tarc-max-k", type=int, default=50)
    parser.add_argument("--baseline-qf", type=int, default=20,
                        help="Query freq of the primary baseline (default: 20, DSRL published value)")
    parser.add_argument("--baseline-qf2", type=int, default=None,
                        help="Optional second (matched-frequency) baseline QF, e.g. 32 = mean TARC K")
    parser.add_argument("--baseline-qf2-val", type=float, default=None,
                        help="Pin the converged success rate shown for baseline-qf2 (overrides dynamic computation)")
    parser.add_argument("--plot-style", choices=["curve", "bar"], default="curve",
                        help="'curve' = training curves + K-dist (default); 'bar' = 3-panel bar chart")
    parser.add_argument("--poster", action="store_true",
                        help="Poster variant of the bar figure (8.0x3.2 in, 24 pt, grey baseline)")
    parser.add_argument("--from-cache", action="store_true",
                        help="Skip W&B; load curves from JSON cache")
    parser.add_argument("--output-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = parser.parse_args()

    qf2_tag = f"_qf2-{args.baseline_qf2}" if args.baseline_qf2 is not None else ""
    cache_path = os.path.join(
        args.output_dir,
        f"DSRL_pi0_matching_data_task{args.task_id}_sc{args.tarc_sc:g}_qf{args.baseline_qf}{qf2_tag}.json",
    )

    if args.from_cache:
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"No cache at {cache_path}; run without --from-cache first.")
        print(f"Loading cached curves from {cache_path}")
        tarc_curves, baseline_curves, baseline2_curves = load_cache(cache_path)
    else:
        print(f"Fetching TARC task={args.task_id} sc={args.tarc_sc} "
              f"K=[{args.tarc_min_k},{args.tarc_max_k}] from {args.tarc_project} ...")
        tarc_curves = fetch_tarc_curves(args.tarc_project,
                                        switch_cost=args.tarc_sc,
                                        min_K=args.tarc_min_k,
                                        max_K=args.tarc_max_k,
                                        task_id=args.task_id)
        print(f"\nFetching baseline task={args.task_id} QF={args.baseline_qf} "
              f"from {args.baseline_project} ...")
        baseline_curves = fetch_baseline_curves(args.baseline_project,
                                                query_freq=args.baseline_qf,
                                                task_id=args.task_id)
        baseline2_curves = None
        if args.baseline_qf2 is not None:
            print(f"\nFetching matched-freq baseline task={args.task_id} QF={args.baseline_qf2} "
                  f"from {args.baseline_project} ...")
            baseline2_curves = fetch_baseline_curves(args.baseline_project,
                                                     query_freq=args.baseline_qf2,
                                                     task_id=args.task_id)
        save_cache(tarc_curves, baseline_curves, cache_path, baseline2_curves=baseline2_curves)

    print(f"\nTARC seeds:      {[d['seed'] for d in tarc_curves]}")
    print(f"Baseline seeds:  {[d['seed'] for d in baseline_curves]}")
    if baseline2_curves:
        print(f"Baseline2 seeds: {[d['seed'] for d in baseline2_curves]}")

    if not tarc_curves or not baseline_curves:
        print("WARN: missing TARC or baseline runs — figure will be incomplete.")

    plot_fn = make_bar_figure if args.plot_style == "bar" else make_figure
    plot_fn(tarc_curves, baseline_curves,
            output_dir=args.output_dir,
            tarc_sc=args.tarc_sc,
            baseline_qf=args.baseline_qf,
            tarc_min_K=args.tarc_min_k,
            tarc_max_K=args.tarc_max_k,
            task_id=args.task_id,
            baseline2_curves=baseline2_curves,
            baseline2_qf=args.baseline_qf2,
            baseline2_final_val=args.baseline_qf2_val,
            **({"poster": True} if args.poster else {}))


if __name__ == "__main__":
    main()
