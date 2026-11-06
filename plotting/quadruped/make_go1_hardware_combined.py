"""Combined Go1 hardware results figure for NeurIPS/CoRL.

Generates a single 1×3 figure:
  (a) Penalized reward           — from hardware CSV files
  (b) Unpenalized reward         — from hardware CSV files
  (c) Avg. control frequency     — from hardware CSV files, with min-freq dashes

Saves Go1_HW_Combined[_tarc_only].{pdf,png} next to this script.

Usage:
    cd plotting/quadruped
    python make_go1_hardware_combined.py [--tarc_only] [--no_latex]
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.lines import Line2D
from tueplots import bundles

HERE        = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, '..', '..', 'results', 'quadruped', 'sim2realResults')

TASKS = {
    'Run Then Turn': 'Fixed Sequence: run_then_turn_20s',
    'Gentle Curve':  'Fixed Sequence: gentle_curve_active_20s',
    'Vel. Steps':    'Fixed Sequence: velocity_step_changes_active_20s',
}
SWITCH_COST = 0.005   # applied uniformly to all agents


def _get_val(data, agent, task, metric, col):
    try:
        return float(data[agent][task][col][metric])
    except (KeyError, TypeError):
        return np.nan if col == 'Mean' else 0.0


FILE_PREFIX = {
    'TARC4':  'TARC4',
    'TARC5':  'Tacos5',
    'TARC10': 'Tacos10',
}


def _load_data(agents):
    """Load hardware CSVs; return {agent: {task: DataFrame[Mean, SEM]}}."""
    data = {}
    for agent in agents:
        agent_data = {}
        for task_key, task_csv in TASKS.items():
            runs = []
            for seed in range(5):
                if agent == 'PPO':
                    path = os.path.join(RESULTS_DIR, 'PPO', f'ppo_seed{seed}.csv')
                else:
                    prefix = FILE_PREFIX.get(agent, agent)
                    path = os.path.join(RESULTS_DIR, 'TARC', agent,
                                        f'seed{seed}', f'{prefix}_seed{seed}.csv')
                if not os.path.exists(path):
                    continue
                try:
                    df  = pd.read_csv(path)
                    row = df[df['ModeDetails'] == task_csv].iloc[:, 2:].mean(axis=0)
                    runs.append(row)
                except Exception:
                    continue
            if not runs:
                continue
            df_s = pd.DataFrame(runs)
            df_s['Avg Control Frequency'] = (df_s['Total Action Switches']
                                             / (df_s['Env Steps'] * 0.02))
            df_s['Penalized Reward']   = (df_s['Total Reward']
                                          - SWITCH_COST * df_s['Total Action Switches'])
            df_s['Unpenalized Reward'] = df_s['Total Reward']
            agent_data[task_key] = pd.DataFrame({'Mean': df_s.mean(), 'SEM': df_s.sem()})
        data[agent] = agent_data
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tarc_only', action='store_true',
                        help='Only Baseline + TARC-4 (default: all variants)')
    parser.add_argument('--no_latex', action='store_true',
                        help='Disable LaTeX rendering (for quick preview)')
    args = parser.parse_args()

    if args.tarc_only:
        agents       = ['PPO', 'TARC4']
        plot_names   = ['Baseline', 'TARC']
        min_freq_map = {'TARC4': 50 / 4}
    else:
        agents       = ['PPO', 'TARC4', 'TARC5', 'TARC10']
        plot_names   = ['Baseline', 'TARC-4', 'TARC-5', 'TARC-10']
        min_freq_map = {'TARC4': 50/4, 'TARC5': 50/5, 'TARC10': 50/10}

    print('Loading hardware CSV data...')
    data      = _load_data(agents)
    task_keys = list(TASKS.keys())

    # ── Style ──────────────────────────────────────────────────────────────
    rc = bundles.neurips2024()
    if args.no_latex:
        rc.pop('text.latex.preamble', None)
        rc['text.usetex'] = False
    plt.rcParams.update(rc)

    task_colors = sns.color_palette('colorblind', n_colors=len(task_keys))

    # ── Figure ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6))

    n   = len(agents)
    x   = np.arange(n)
    bw  = 0.22
    ekw = {'elinewidth': 0.7, 'markeredgewidth': 0.7, 'ecolor': 'black'}

    panels = [
        #('Penalized Reward',      '(a) Penalized Reward',       'Total Reward'),
        ('Unpenalized Reward',    '(b) Unpenalized Reward',     'Total Reward'),
        ('Avg Control Frequency', '(c) Avg. Control Frequency', 'Frequency (Hz) $\\downarrow$ better'),
    ]

    for ax, (metric, title, ylabel) in zip(axes, panels):
        for j, (task_key, color) in enumerate(zip(task_keys, task_colors)):
            means  = [_get_val(data, ag, task_key, metric, 'Mean') for ag in agents]
            sems   = [_get_val(data, ag, task_key, metric, 'SEM')  for ag in agents]
            offset = bw * (j - (len(task_keys) - 1) / 2)
            ax.bar(x + offset, means, bw, yerr=sems, color=color, capsize=2,
                   error_kw=ekw, zorder=3)

        if metric == 'Avg Control Frequency':
            for i, ag in enumerate(agents):
                if ag in min_freq_map:
                    hw = bw * len(task_keys) / 2 + 0.02
                    ax.hlines(min_freq_map[ag], i - hw, i + hw,
                              colors='black', lw=0.9, linestyles='dashed', zorder=4)

        title = 'Performance Comparison' if title == '(b) Unpenalized Reward' else title
        ylabel = 'Task Performance' if ylabel == 'Total Reward' else ylabel
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_names, rotation=30, ha='right')
        ax.grid(True, axis='y', alpha=0.35, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        y_lo, y_hi = ax.get_ylim()
        ax.set_ylim(bottom=min(0, y_lo), top=y_hi * 1.15)

    # ── Legend: task colors + min-freq dash ────────────────────────────────
    lfs = plt.rcParams.get('legend.fontsize', 7)
    patch_handles = [mpatches.Patch(color=task_colors[j], label=k)
                     for j, k in enumerate(task_keys)]
    dash_handle   = Line2D([0], [0], color='black', lw=0.9, ls='--',
                            label='Min. freq. (TARC)')
    all_handles   = patch_handles + [dash_handle]

    fig.legend(handles=all_handles, loc='upper center',
               bbox_to_anchor=(0.5, 1.01),
               ncol=len(all_handles), frameon=False, fontsize=lfs)

    fig.tight_layout(rect=[0, 0, 1, 0.87])

    # ── Save ───────────────────────────────────────────────────────────────
    suffix = '_tarc_only' if args.tarc_only else ''
    out = os.path.join(HERE, f'Go1_HW_Combined{suffix}.pdf')
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
