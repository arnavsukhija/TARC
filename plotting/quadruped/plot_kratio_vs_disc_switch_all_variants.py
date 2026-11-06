"""Plot K_ratio (target) vs realised disc_switch_ratio — all TARC variants.

Extends plot_kratio_vs_disc_switch.py to show TARC-4, TARC-5, TARC-10
on the same axes, intended for the appendix.

X-axis: K_ratio (target discounted query ratio).
Y-axis: (a) realised disc_switch_ratio, (b) total reward.
"""

import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots  # noqa: F401
import wandb


ENTITY = 'arnavsukhija-eth-zurich'

SOURCES = [
    ('Go1_AutoLambda_Eval', [4, 5, 10]),
]

BASELINE_EVAL_PROJECT  = 'Go1_Stability_Rebuttal_Eval'
BASELINE_TRAIN_PROJECT = 'Go1_Stability_Rebuttal'
BASELINE_LABEL         = 'PPO-50Hz'
BASELINE_REQUIRE = {
    'switch_cost_wrapper': False,
    'num_timesteps': 200_000_000,
    'control_frequency_hz': 50.0,
}

MAX_TIME_REPEATS = [4, 5, 10]
TARC_NAMES       = {4: 'TARC-4', 5: 'TARC-5', 10: 'TARC-10'}
N_EVAL_ROLLOUTS  = 10  # Results_0_random … Results_9_random

# Per-variant K_ratio whitelist; None means accept all K_ratios.
K_RATIO_WHITELIST = {
    5: {0.3, 0.4, 0.6, 0.7},
}

METRICS = {
    'disc_switch_ratio': 'Results_{i}_random/disc_switch_ratio',
    'reward':            'Results_{i}_random/Total reward',
}


def fetch_eval_data():
    api = wandb.Api()
    samples = {m: defaultdict(list) for m in METRICS}

    for project, allowed_mtrs in SOURCES:
        print(f'  fetching from {ENTITY}/{project} (mtr in {allowed_mtrs}) ...')
        runs = api.runs(f'{ENTITY}/{project}')
        for run in runs:
            cfg     = run.config
            K_ratio = cfg.get('K_ratio')
            mtr     = cfg.get('max_time_repeat')
            if K_ratio is None or mtr is None:
                continue
            if int(mtr) not in allowed_mtrs:
                continue

            whitelist = K_RATIO_WHITELIST.get(int(mtr))
            if whitelist is not None and round(float(K_ratio), 6) not in {round(k, 6) for k in whitelist}:
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


def fetch_baseline_rewards():
    """Return flat list of Total reward values for the PPO-50Hz baseline."""
    api = wandb.Api()

    print(f'  scanning {ENTITY}/{BASELINE_TRAIN_PROJECT} for baseline training runs ...')
    try:
        train_runs = list(api.runs(f'{ENTITY}/{BASELINE_TRAIN_PROJECT}'))
    except Exception as e:
        print(f'  could not fetch training runs: {e}')
        return []

    matching_train_ids = set()
    for tr in train_runs:
        cfg = tr.config
        if all(cfg.get(k) == v for k, v in BASELINE_REQUIRE.items()):
            matching_train_ids.add(tr.id)
    print(f'    {len(matching_train_ids)} training runs match baseline filter '
          f'({BASELINE_REQUIRE})')
    if not matching_train_ids:
        return []

    print(f'  pulling rewards from {ENTITY}/{BASELINE_EVAL_PROJECT} ...')
    try:
        eval_runs = list(api.runs(f'{ENTITY}/{BASELINE_EVAL_PROJECT}'))
    except Exception as e:
        print(f'  could not fetch eval runs: {e}')
        return []

    rewards = []
    matched_train_ids = set()
    for run in eval_runs:
        # training_go1.py sets name=wandb_run_id for eval runs
        train_id = run.name
        if train_id not in matching_train_ids:
            continue
        try:
            summary = run.summary._json_dict
        except Exception:
            summary = dict(run.summary)
        added = 0
        for i in range(N_EVAL_ROLLOUTS):
            val = summary.get(f'Results_{i}_random/Total reward')
            if val is None:
                continue
            rewards.append(float(val))
            added += 1
        if added:
            matched_train_ids.add(train_id)

    missing = matching_train_ids - matched_train_ids
    if missing:
        print(f'    {len(missing)} matching training runs had no eval run yet '
              f'(first few: {list(sorted(missing))[:5]})')
    print(f'    {len(matched_train_ids)} eval runs matched, {len(rewards)} reward samples')
    return rewards


def aggregate(metric_samples):
    """Return {mtr: (K_ratios, means, sems, counts)} sorted by K_ratio."""
    out = {}
    for mtr in MAX_TIME_REPEATS:
        rows = [(k, v) for (k, m), v in metric_samples.items() if m == mtr]
        rows.sort(key=lambda kv: kv[0])
        if not rows:
            continue
        ks     = np.array([k for k, _ in rows])
        means  = np.array([np.mean(v) for _, v in rows])
        sems   = np.array([np.std(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
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

    ax.set_xlabel(r'Target discounted query ratio $K$')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, lw=0.4)


def _overlay_baseline(ax, rewards):
    if not rewards:
        return
    arr  = np.asarray(rewards, dtype=float)
    mean = float(np.mean(arr))
    sem  = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    lo, hi = ax.get_xlim()
    ax.axhspan(mean - sem, mean + sem, color='gray', alpha=0.15, zorder=0)
    ax.axhline(mean, color='black', lw=1.0, linestyle='--', alpha=0.8,
               label=f'{BASELINE_LABEL} (no querying constraint, $K=1$)')
    ax.set_xlim(lo, hi)
    print(f'\n=== Baseline overlay ===\n{BASELINE_LABEL}: n={len(arr)}  '
          f'mean={mean:.4f}  sem={sem:.4f}')


def create_plot(agg_switch, agg_reward, baseline_rewards):
    plt.style.use(['science', 'no-latex'])
    base_colors = sns.color_palette('colorblind')
    color_map   = {mtr: base_colors[i] for i, mtr in enumerate(MAX_TIME_REPEATS)}

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))

    _plot_curves(axes[0], agg_switch, color_map,
                 ylabel=r'Realised discounted query ratio',
                 title='(a) Constraint satisfaction',
                 identity=True)
    _plot_curves(axes[1], agg_reward, color_map,
                 ylabel='Total reward',
                 title='(b) Task performance')
    _overlay_baseline(axes[1], baseline_rewards)

    handles_a, labels_a = axes[0].get_legend_handles_labels()
    handles_b, labels_b = axes[1].get_legend_handles_labels()
    seen, handles, labels = set(), [], []
    for h, l in list(zip(handles_a, labels_a)) + list(zip(handles_b, labels_b)):
        if l in seen:
            continue
        seen.add(l)
        handles.append(h)
        labels.append(l)
    fig.legend(handles=handles, labels=labels,
               loc='upper center', bbox_to_anchor=(0.5, 1.06),
               ncol=len(labels), frameon=False, fontsize='small')

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_pdf = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'Go1_KRatio_vs_DiscSwitchRatio_AllVariants.pdf')
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.replace('.pdf', '.png'), dpi=300)
    print(f'Plot saved: {out_pdf}')
    return out_pdf


def main():
    print('Fetching eval runs ...')
    samples = fetch_eval_data()
    if not any(samples.values()):
        print('No samples found. Check SOURCES and that runs have logged '
              'Results_{i}_random/disc_switch_ratio and /Total reward keys.')
        return

    baseline_rewards = fetch_baseline_rewards()

    agg_switch = aggregate(samples['disc_switch_ratio'])
    agg_reward = aggregate(samples['reward'])

    for label, agg in [('disc_switch_ratio', agg_switch), ('reward', agg_reward)]:
        print(f'\n=== Aggregated stats: {label} ===')
        print(f'{"Variant":<10} {"K_ratio":<8} {"n":<5} {"mean":<10} {"sem":<10}')
        for mtr in MAX_TIME_REPEATS:
            if mtr not in agg:
                continue
            ks, means, sems, counts = agg[mtr]
            for k, m, s, n in zip(ks, means, sems, counts):
                print(f'{TARC_NAMES[mtr]:<10} {k:<8.3f} {n:<5d} {m:<10.4f} {s:<10.4f}')

    create_plot(agg_switch, agg_reward, baseline_rewards)


if __name__ == '__main__':
    main()
