"""Plot K_ratio (target) vs realised disc_switch_ratio for Go1 TARC-{5,10}.

Pulls eval-project runs from W&B (Go1_AutoLambda_Eval), groups by
(K_ratio, max_time_repeat), and aggregates over (training seeds x eval seeds)
to produce mean +/- standard-error curves.

X-axis: K_ratio (target discounted switch ratio).
Y-axis: realised disc_switch_ratio averaged over rollouts.
One curve per max_time_repeat (TARC-5, TARC-10).
"""

import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots  # noqa: F401  (registers matplotlib style)
import wandb


ENTITY = 'arnavsukhija-eth-zurich'
PROJECT = 'Go1_AutoLambda_Eval'

MAX_TIME_REPEATS = [5, 10]
TARC_NAMES = {5: 'TARC-5', 10: 'TARC-10'}
N_EVAL_ROLLOUTS = 10  # Results_0_random ... Results_9_random


METRICS = {
    'disc_switch_ratio': 'Results_{i}_random/disc_switch_ratio',
    'reward':            'Results_{i}_random/Total reward',
}


def fetch_eval_data():
    """Return dict[metric][(K_ratio, max_time_repeat)] -> list of per-rollout values.

    Each list contains every Results_{i}_random/<metric> across all matching
    runs (5 training seeds * 10 rollouts = 50 samples per cell).
    """
    api = wandb.Api()
    runs = api.runs(f'{ENTITY}/{PROJECT}')

    samples = {m: defaultdict(list) for m in METRICS}
    for run in runs:
        cfg = run.config
        K_ratio = cfg.get('K_ratio')
        mtr = cfg.get('max_time_repeat')
        if K_ratio is None or mtr is None:
            continue
        if int(mtr) not in MAX_TIME_REPEATS:
            continue

        try:
            summary = run.summary._json_dict
        except Exception:
            summary = dict(run.summary)

        cell = (float(K_ratio), int(mtr))
        for i in range(N_EVAL_ROLLOUTS):
            for metric, key_tmpl in METRICS.items():
                val = summary.get(key_tmpl.format(i=i))
                if val is None:
                    continue
                samples[metric][cell].append(float(val))

    return samples


def aggregate(metric_samples):
    """Return {mtr: (K_ratios, means, sems, counts)} sorted by K_ratio."""
    out = {}
    for mtr in MAX_TIME_REPEATS:
        rows = [(k, v) for (k, m), v in metric_samples.items() if m == mtr]
        rows.sort(key=lambda kv: kv[0])
        if not rows:
            continue
        ks = np.array([k for k, _ in rows])
        means = np.array([np.mean(v) for _, v in rows])
        sems = np.array([np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
                         for _, v in rows])
        counts = np.array([len(v) for _, v in rows])
        out[mtr] = (ks, means, sems, counts)
    return out


def _plot_curves(ax, agg, color_map, ylabel, title, identity=False):
    if identity:
        all_ks = sorted({float(k) for ks, *_ in agg.values() for k in ks})
        if all_ks:
            lo, hi = all_ks[0], all_ks[-1]
            ax.plot([lo, hi], [lo, hi], linestyle='--', color='black',
                    lw=0.8, alpha=0.6, label='Target ($y = K$)')

    for mtr in MAX_TIME_REPEATS:
        if mtr not in agg:
            continue
        ks, means, sems, _ = agg[mtr]
        color = color_map[mtr]
        ax.plot(ks, means, marker='o', lw=1.2, color=color, label=TARC_NAMES[mtr])
        ax.fill_between(ks, means - sems, means + sems, alpha=0.2, color=color)

    ax.set_xlabel(r'Target discounted switch ratio $K$')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, lw=0.4)


def create_plot(agg_switch, agg_reward):
    plt.style.use(['science', 'no-latex'])
    base_colors = sns.color_palette("colorblind")
    color_map = {5: base_colors[1], 10: base_colors[2]}

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.6))

    _plot_curves(axes[0], agg_switch, color_map,
                 ylabel=r'Realised discounted switch ratio',
                 title='(a) Constraint satisfaction',
                 identity=True)
    _plot_curves(axes[1], agg_reward, color_map,
                 ylabel='Total reward',
                 title='(b) Task performance')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels,
               loc='upper center', bbox_to_anchor=(0.5, 1.04),
               ncol=len(labels), frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_pdf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'Go1_KRatio_vs_DiscSwitchRatio.pdf')
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.replace('.pdf', '.png'), dpi=300)
    print(f'Plot saved: {out_pdf}')
    return out_pdf


def main():
    print(f'Fetching eval runs from {ENTITY}/{PROJECT} ...')
    samples = fetch_eval_data()
    if not any(samples.values()):
        print('No samples found. Check ENTITY/PROJECT and that runs have logged '
              'Results_{i}_random/disc_switch_ratio and /Total reward keys.')
        return

    agg_switch = aggregate(samples['disc_switch_ratio'])
    agg_reward = aggregate(samples['reward'])

    for label, agg in [('disc_switch_ratio', agg_switch), ('reward', agg_reward)]:
        print(f'\n=== Aggregated stats: {label} ===')
        print(f'{"TARC":<8} {"K_ratio":<8} {"n":<5} {"mean":<10} {"sem":<10}')
        for mtr in MAX_TIME_REPEATS:
            if mtr not in agg:
                continue
            ks, means, sems, counts = agg[mtr]
            for k, m, s, n in zip(ks, means, sems, counts):
                print(f'{TARC_NAMES[mtr]:<8} {k:<8.3f} {n:<5d} {m:<10.4f} {s:<10.4f}')

    create_plot(agg_switch, agg_reward)


if __name__ == '__main__':
    main()
