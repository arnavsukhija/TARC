"""Diagnose the min_K=1 K-attractor: pull K within-episode stats from finished runs.

Question: is avg_K=22 a unimodal collapse at K=22, or a bimodal K=1 + K=large mix?

The training already logs (per eval call, averaged over rollouts):
  evaluation/avg_K_per_decision         -> mean K
  evaluation/avg_K_std_within_episode   -> std of K within a rollout
  evaluation/avg_K_min_within_episode   -> min K seen in a rollout
  evaluation/avg_K_max_within_episode   -> max K seen in a rollout

Bimodal collapse signature (the buffer-bias hypothesis):
  - K_min == 1  consistently
  - K_max ~ k_max (e.g. 50)
  - K_std large (~ half the range)

Unimodal collapse signature (gradient/exploration problem):
  - K_min, K_max close to mean
  - K_std small
"""
import argparse
from collections import defaultdict

N_LAST_EVAL = 5
MIN_STEPS = 400_000


def fetch_k_stats(project):
    import wandb
    api = wandb.Api(timeout=29)
    runs = api.runs(project)

    rows = []
    for run in runs:
        c = run.config
        if not c.get("switch_cost_wrapper"):
            continue
        sc = c.get("switch_cost")
        min_K = c.get("min_time_between_switches")
        bias = float(c.get("pseudo_time_init_bias", 0.0) or 0.0)
        eval_mode = c.get("eval_mode", "hybrid")
        seed = c.get("seed", -1)
        if sc is None or min_K is None:
            continue

        history = run.history(
            keys=[
                "evaluation/success_rate",
                "evaluation/avg_K_per_decision",
                "evaluation/avg_K_std_within_episode",
                "evaluation/avg_K_min_within_episode",
                "evaluation/avg_K_max_within_episode",
                "_step",
            ],
            samples=2000,
        )
        if history.empty:
            continue
        # Use last N_LAST_EVAL evals
        h = history.dropna(subset=["evaluation/avg_K_per_decision"]).tail(N_LAST_EVAL)
        if h.empty or h["_step"].max() < MIN_STEPS:
            continue
        rows.append({
            "sc": sc,
            "minK": min_K,
            "bias": bias,
            "eval": eval_mode,
            "seed": seed,
            "succ": float(h["evaluation/success_rate"].mean()),
            "K_mean": float(h["evaluation/avg_K_per_decision"].mean()),
            "K_std": float(h["evaluation/avg_K_std_within_episode"].mean()),
            "K_min": float(h["evaluation/avg_K_min_within_episode"].mean()),
            "K_max": float(h["evaluation/avg_K_max_within_episode"].mean()),
        })
    return rows


def summarize(rows):
    """Group by (sc, minK, bias, eval) and aggregate over seeds."""
    g = defaultdict(list)
    for r in rows:
        key = (r["sc"], r["minK"], r["bias"], r["eval"])
        g[key].append(r)

    print(f"{'sc':>4} {'minK':>4} {'bias':>4} {'eval':<20} {'n':>2} "
          f"{'succ':>14} {'K_mean':>14} {'K_std':>14} {'K_min':>10} {'K_max':>10}")
    print("-" * 110)
    for key in sorted(g.keys()):
        sc, mK, b, ev = key
        seeds = g[key]
        n = len(seeds)
        def ms(field):
            xs = [s[field] for s in seeds]
            import numpy as np
            return float(np.mean(xs)), float(np.std(xs, ddof=1) / max(1, n ** 0.5)) if n > 1 else 0.0
        succ_m, succ_s = ms("succ")
        Km_m, Km_s = ms("K_mean")
        Ks_m, Ks_s = ms("K_std")
        Ki_m, _ = ms("K_min")
        Ka_m, _ = ms("K_max")
        print(f"{sc:>4} {mK:>4} {b:>4} {ev:<20} {n:>2} "
              f"  {succ_m:5.3f}±{succ_s:5.3f}   {Km_m:5.2f}±{Km_s:4.2f}   "
              f"{Ks_m:5.2f}±{Ks_s:4.2f}   {Ki_m:5.2f}   {Ka_m:5.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation", default="DSRL_pi0_Libero_TARC_ablation")
    args = p.parse_args()
    print(f"Fetching K stats from {args.ablation} ...")
    rows = fetch_k_stats(args.ablation)
    summarize(rows)


if __name__ == "__main__":
    main()
