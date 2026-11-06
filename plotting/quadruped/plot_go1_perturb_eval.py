"""Perturbation-eval comparison: TARC-4/5/10 vs PPO baselines at 50/20/12.5 Hz.

Same structure as plot_go1_results.py — fetches Results_{i}_random/ summary keys
from a W&B eval project and produces three bar-chart panels.

Usage:
    cd plotting/quadruped
    python plot_go1_perturb_eval.py [--project perturbEval] [--tarc_only]
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import scienceplots  # noqa: F401
import wandb

plt.style.use(['science', 'no-latex'])
plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.8, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
})

ENTITY = 'arnavsukhija-eth-zurich'

N_ROLLOUTS   = 10
BASE_FREQ    = 50.0
BASE_EP_LEN  = 1000
BASE_CTRL_DT = 0.02
SWITCH_COST  = 0.005

ORDER = ['TARC-4', 'TARC-5', 'TARC-10', 'PPO-50Hz', 'PPO-20Hz', 'PPO-12.5Hz']

LABELS = {
    'TARC-4':     'TARC-4',
    'TARC-5':     'TARC-5',
    'TARC-10':    'TARC-10',
    'PPO-50Hz':   'PPO (50 Hz)',
    'PPO-20Hz':   'PPO (20 Hz)',
    'PPO-12.5Hz': 'PPO (12.5 Hz)',
}

base_colors = sns.color_palette('colorblind')
COLOR_MAP = {
    'TARC-4':     base_colors[1],
    'TARC-5':     base_colors[1],
    'TARC-10':    base_colors[1],
    'PPO-50Hz':   base_colors[0],
    'PPO-20Hz':   base_colors[0],
    'PPO-12.5Hz': base_colors[0],
}
HATCH_MAP = {
    'TARC-4':     '',
    'TARC-5':     '..',
    'TARC-10':    '//',
    'PPO-50Hz':   '',
    'PPO-20Hz':   '..',
    'PPO-12.5Hz': '//',
}
EDGE_MAP = {m: base_colors[1] if 'TARC' in m else base_colors[0] for m in ORDER}

MIN_FREQ_MAP = {
    'TARC-4':  BASE_FREQ / 4,
    'TARC-5':  BASE_FREQ / 5,
    'TARC-10': BASE_FREQ / 10,
}


def identify_method(cfg):
    is_tarc = bool(cfg.get('switch_cost_wrapper', False))
    mtr     = cfg.get('max_time_repeat')
    divisor = float(cfg.get('base_dt_divisor', 1))
    if is_tarc and mtr is not None and int(mtr) in (4, 5, 10):
        return f'TARC-{int(mtr)}'
    if not is_tarc:
        if abs(divisor - 1.0) < 1e-6:
            return 'PPO-50Hz'
        if abs(divisor - 2.5) < 1e-6:
            return 'PPO-20Hz'
        if abs(divisor - 4.0) < 1e-6:
            return 'PPO-12.5Hz'
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=str, default='perturbEval',
                        help='W&B eval project (default: perturbEval)')
    parser.add_argument('--tarc_only', action='store_true',
                        help='Show only TARC-5 (as "TARC") vs PPO baselines')
    parser.add_argument('--ppo_only', action="store_true",
                         help='Show only PPO baselines (50/20 Hz)')
    args = parser.parse_args()

    # ── Fetch ──────────────────────────────────────────────────────────────
    api = wandb.Api()
    print(f'Fetching runs from {ENTITY}/{args.project} ...')
    runs = list(api.runs(f'{ENTITY}/{args.project}'))
    print(f'  Total runs: {len(runs)}')

    records = []
    for run in runs:
        cfg    = run.config
        method = identify_method(cfg)
        if method is None:
            continue

        try:
            summary = run.summary._json_dict
        except Exception:
            summary = dict(run.summary)

        divisor = float(cfg.get('base_dt_divisor', 1))
        ep_len  = int(round(BASE_EP_LEN / divisor))
        ctrl_dt = BASE_CTRL_DT * divisor

        for i in range(N_ROLLOUTS):
            reward      = summary.get(f'Results_{i}_random/Total reward')
            n_act       = summary.get(f'Results_{i}_random/Number of actions')
            avg_freq    = summary.get(f'Results_{i}_random/Avg control frequency (Hz)')
            median_freq = summary.get(f'Results_{i}_random/Median control frequency (Hz)')

            if reward is None:
                continue
            n_act_val    = float(n_act) if n_act is not None else float(ep_len)
            fallback_avg = n_act_val / (ep_len * ctrl_dt)

            freq_avg = float(avg_freq) if avg_freq is not None else (
                       float(median_freq) if median_freq is not None else fallback_avg)
            freq_med = float(median_freq) if median_freq is not None else freq_avg

            records.append({
                'Method':          method,
                'Run':             run.id,
                'Seed':            i,
                'Reward':          float(reward),
                'PenalizedReward': float(reward) - SWITCH_COST * n_act_val,
                'Freq':            freq_avg,
                'MedianFreq':      freq_med,
            })

    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError(f'No matching runs found in {ENTITY}/{args.project}.')

    print(f'\nLoaded {len(df)} rollouts from {df["Run"].nunique()} eval runs:')
    print(df.groupby('Method')[['Reward', 'PenalizedReward', 'MedianFreq']]
            .agg(['count', 'mean', 'sem']).to_string())

    # ── Plot ───────────────────────────────────────────────────────────────
    if args.tarc_only:
        active_order = ['TARC-5', 'PPO-50Hz', 'PPO-20Hz', 'PPO-12.5Hz']
        # rename TARC-5 → TARC in the data so groupby/labels work uniformly
        df = df.copy()
        df['Method'] = df['Method'].replace({'TARC-5': 'TARC'})
        active_order = ['TARC', 'PPO-50Hz', 'PPO-20Hz', 'PPO-12.5Hz']
        color_map  = {'TARC': base_colors[1], 'PPO-50Hz': base_colors[0],
                      'PPO-20Hz': base_colors[2], 'PPO-12.5Hz': base_colors[0]}
        hatch_map  = {'TARC': '', 'PPO-50Hz': '', 'PPO-20Hz': '', 'PPO-12.5Hz': '//'}
        edge_map   = {'TARC': base_colors[1], 'PPO-50Hz': base_colors[0],
                      'PPO-20Hz': base_colors[2], 'PPO-12.5Hz': base_colors[0]}
        min_freq_map = {'TARC': BASE_FREQ / 5}
        display_labels = {'TARC': 'TARC (ours)\nadaptive, avg. $\\approx$20 Hz', 
                          'PPO-50Hz': 'Fixed rate\n50 Hz',
                          'PPO-20Hz': 'Fixed rate\n20 Hz', 'PPO-12.5Hz': 'PPO (12.5 Hz)'}
    elif args.ppo_only:
        active_order = ['PPO-50Hz', 'PPO-20Hz', 'PPO-12.5Hz']
        color_map  = {m: COLOR_MAP[m] for m in active_order}
        hatch_map  = {m: HATCH_MAP[m] for m in active_order}
        edge_map   = {m: EDGE_MAP[m] for m in active_order}
        min_freq_map = {}
        display_labels = {'PPO-50Hz': '50 Hz',
                          'PPO-20Hz': '20 Hz', 'PPO-12.5Hz': '12.5 Hz'}
    else:
        active_order   = ORDER
        color_map      = COLOR_MAP
        hatch_map      = HATCH_MAP
        edge_map       = EDGE_MAP
        min_freq_map   = MIN_FREQ_MAP
        display_labels = LABELS

    present = [m for m in active_order if m in df['Method'].unique()]

    fig, ax = plt.subplots(figsize=(3.6, 3.0))

    panels = [
        #('PenalizedReward', r'(a) Penalized Reward',         'Total Reward'),
        ('Reward',          r'Performance Comparison',       'Task Reward'),
        #('Freq',             r'Avg. Control Frequency',   'Frequency (Hz)'),
    ]

    for (col, title, ylabel) in panels:
        g = df.groupby('Method')[col]
        centers  = g.mean().reindex(present)
        iqr_half = g.sem().reindex(present)

        valid = [m for m in present if not np.isnan(centers[m])]
        x     = np.arange(len(valid))

        bars = ax.bar(
            x,
            centers[valid].values,
            yerr=iqr_half[valid].values,
            capsize=3,
            width=0.55,
            color=[color_map[m] for m in valid],
            error_kw={'lw': 1.0, 'markeredgewidth': 1.0, 'ecolor': 'black'},
            zorder=3,
        )
        for bar, m in zip(bars, valid):
            bar.set_hatch(hatch_map[m])
            bar.set_edgecolor('none' if not hatch_map[m] else edge_map[m])
            bar.set_linewidth(1.2)

        if col == 'Freq':
            for j, m in enumerate(valid):
                if m in min_freq_map:
                    ax.hlines(min_freq_map[m], j - 0.28, j + 0.28,
                              colors='black', linestyles='dashed', lw=0.9, zorder=4)

        #ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([display_labels[m] for m in valid], rotation=0, ha='center', fontsize=7)
        ax.grid(True, axis='y', alpha=0.4, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        y_min, y_max = ax.get_ylim()
        bottom = 20 if col in ('PenalizedReward', 'Reward') else min(0, y_min)
        ax.set_ylim(bottom=bottom, top=y_max * 1.12)

    # ── Legend ─────────────────────────────────────────────────────────────
    if args.tarc_only:
        _legend_spec = [
            ('TARC (ours)\nadaptive, avg. $\\approx$20 Hz', dict(facecolor=base_colors[1]), 'TARC'),
            ('Fixed rate\n50 Hz',                           dict(facecolor=base_colors[0]), 'PPO-50 Hz'),
            ('Fixed rate\n20 Hz',                           dict(facecolor=base_colors[2]), 'PPO-20 Hz'),
            ('Fixed rate\n12.5 Hz',                         dict(facecolor=base_colors[0], hatch='//', edgecolor=base_colors[0]), 'PPO-12.5 Hz')]
    elif args.ppo_only:
        _legend_spec = [
            ('50Hz',   dict(facecolor=COLOR_MAP['PPO-50Hz']),                                                         '50 Hz'),
            ('20Hz',   dict(facecolor=COLOR_MAP['PPO-20Hz'],  hatch='..', edgecolor=EDGE_MAP['PPO-20Hz']),            '20 Hz'),
            ('12.5Hz', dict(facecolor=COLOR_MAP['PPO-12.5Hz'], hatch='//', edgecolor=EDGE_MAP['PPO-12.5Hz']),         '12.5 Hz'),
        ]
    else:
        _legend_spec = [
            ('TARC-4',     dict(facecolor=COLOR_MAP['TARC-4']),                                                          'TARC-4'),
            ('TARC-5',     dict(facecolor=COLOR_MAP['TARC-5'],    hatch='..', edgecolor=EDGE_MAP['TARC-5']),              'TARC-5'),
            ('TARC-10',    dict(facecolor=COLOR_MAP['TARC-10'],   hatch='//', edgecolor=EDGE_MAP['TARC-10']),             'TARC-10'),
            ('PPO-50Hz',   dict(facecolor=COLOR_MAP['PPO-50Hz']),                                                         'PPO-50 Hz'),
            ('PPO-20Hz',   dict(facecolor=COLOR_MAP['PPO-20Hz'],  hatch='..', edgecolor=EDGE_MAP['PPO-20Hz']),            'PPO-20 Hz'),
            ('PPO-12.5Hz', dict(facecolor=COLOR_MAP['PPO-12.5Hz'], hatch='//', edgecolor=EDGE_MAP['PPO-12.5Hz']),         'PPO-12.5 Hz'),
        ]
    present_set    = set(df['Method'].unique())
    legend_handles = [mpatches.Patch(**kw, label=lbl)
                      for m, kw, lbl in _legend_spec if m in present_set]

    ncol = 4 if args.tarc_only else 4
    fig.legend(handles=legend_handles,
               loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=ncol, frameon=False, fontsize=8)

    fig.tight_layout()

    out_dir    = os.path.dirname(os.path.abspath(__file__))
    suffix     = '_tarc_only' if args.tarc_only else '_ppo_only' if args.ppo_only else ''
    out_pdf    = os.path.join(out_dir, f'Go1_PerturbEval_Comparison{suffix}.pdf')
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_pdf.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f'\nSaved: {out_pdf}')


if __name__ == '__main__':
    main()
