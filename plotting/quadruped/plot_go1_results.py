"""Low-frequency comparison: TARC-4 vs PPO-50Hz / 20Hz / 16.67Hz / 12.5Hz.

Data: Go1_Stability_Rebuttal_Eval — eval-mode runs, 10 random rollouts per seed
      (Results_{i}_random/... keys, i in 0-9, same seeds across all methods).
Methods identified by config fields in the eval project.
"""

import os
import sys
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

ENTITY       = 'arnavsukhija-eth-zurich'
EVAL_PROJECT = 'Go1_Stability_Rebuttal_Eval'
N_ROLLOUTS   = 10    # Results_0_random … Results_9_random

# Base env: 50 Hz, ctrl_dt=0.02 s, episode_length=1000 steps
BASE_FREQ    = 50.0
BASE_EP_LEN  = 1000
BASE_CTRL_DT = 0.02
SWITCH_COST  = 0.005   # penalty per action applied uniformly for fair comparison

ORDER = ['TARC-4', 'PPO-50Hz', 'PPO-20Hz', 'PPO-12.5Hz']
LABELS = {
    'TARC-4':     'TARC-4',
    'PPO-50Hz':   'PPO (50 Hz)',
    'PPO-20Hz':   'PPO (20 Hz)',
    'PPO-12.5Hz': 'PPO (12.5 Hz)',
}

base_colors = sns.color_palette('colorblind')
COLOR_MAP = {
    'TARC-4':     base_colors[1],   # orange
    'PPO-50Hz':   base_colors[0],   # blue
    'PPO-20Hz':   base_colors[0],
    'PPO-12.5Hz': base_colors[0],
}
HATCH_MAP = {
    'TARC-4':     '',
    'PPO-50Hz':   '',
    'PPO-20Hz':   '..',
    'PPO-12.5Hz': '//',
}
EDGE_MAP = {
    'TARC-4':     base_colors[1],
    'PPO-50Hz':   base_colors[0],
    'PPO-20Hz':   base_colors[0],
    'PPO-12.5Hz': base_colors[0],
}


NTS_50HZ = 600_000_000   # only accept the 600M-step 50Hz baseline


def identify_method(cfg):
    is_tarc = bool(cfg.get('switch_cost_wrapper', False))
    mtr     = cfg.get('max_time_repeat')
    divisor = float(cfg.get('base_dt_divisor', 1))
    nts     = cfg.get('num_timesteps')
    if is_tarc and mtr is not None and int(mtr) == 4:
        return 'TARC-4'
    if not is_tarc and abs(divisor - 1.0) < 1e-6:
        if nts is not None and int(nts) != NTS_50HZ:
            return None   # wrong training length
        return 'PPO-50Hz'
    if not is_tarc and abs(divisor - 2.5) < 1e-6:
        return 'PPO-20Hz'
    if not is_tarc and abs(divisor - 4.0) < 1e-6:
        return 'PPO-12.5Hz'
    return None


# ── Fetch data ─────────────────────────────────────────────────────────────

api = wandb.Api()
print(f'Fetching finished runs from {ENTITY}/{EVAL_PROJECT} ...')
runs = list(api.runs(f'{ENTITY}/{EVAL_PROJECT}'))
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
        # MedianFreq: per-rollout median hold → freq; fall back to avg if not logged
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
    raise RuntimeError(
        f'No data found in {EVAL_PROJECT}. '
        'Run the eval launcher for TARC-4, PPO-25Hz, and PPO-12.5Hz first.'
    )

print(f'\nLoaded {len(df)} rollouts from {df["Run"].nunique()} eval runs:')
print(df.groupby('Method')[['Reward', 'PenalizedReward', 'Freq', 'MedianFreq']].agg(['count', 'mean', 'median', 'sem']).to_string())


# ── Poster variant (6.0 x 3.2 in, single bar chart) ────────────────────────

if '--poster' in sys.argv:
    from go1_poster import make_go1_poster_figure
    make_go1_poster_figure(df, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            'Go1_Sim_Poster.pdf'))
    sys.exit(0)


# ── Plot ───────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(5.5, 2.4))

panels = [
    ('PenalizedReward', r'(a) Penalized Reward',         'Total Reward'),
    ('Reward',          r'(b) Unpenalized Reward',       'Total Reward'),
    ('MedianFreq',      r'(c) Median Control Frequency', 'Frequency (Hz)'),
]

for ax, (col, title, ylabel) in zip(axes, panels):
    g = df.groupby('Method')[col]
    if col == 'MedianFreq':
        # median of per-rollout median freqs; IQR/2 as spread for TARC
        centers = g.median().reindex(ORDER)
        iqr_half = g.apply(lambda s: (s.quantile(0.75) - s.quantile(0.25)) / 2).reindex(ORDER)
    else:
        centers  = g.mean().reindex(ORDER)
        iqr_half = g.sem().reindex(ORDER)   # reused name; holds SEM for reward panels
    valid = [m for m in ORDER if not np.isnan(centers[m])]
    x     = np.arange(len(valid))

    bars = ax.bar(
        x,
        centers[valid].values,
        yerr=iqr_half[valid].values if col != 'MedianFreq' else None,
        capsize=3,
        width=0.55,
        color=[COLOR_MAP[m] for m in valid],
        error_kw={'lw': 1.0, 'markeredgewidth': 1.0, 'ecolor': 'black'},
        zorder=3,
    )
    for bar, m in zip(bars, valid):
        bar.set_hatch(HATCH_MAP[m])
        bar.set_edgecolor('none' if not HATCH_MAP[m] else EDGE_MAP[m])
        bar.set_linewidth(1.2)

    # Freq panel: IQR/2 error bar only for TARC (baselines have fixed frequency)
    if col == 'MedianFreq':
        for j, m in enumerate(valid):
            if 'TARC' in m:
                ax.errorbar(j, centers[m], yerr=iqr_half[m], fmt='none',
                            capsize=3, ecolor='black', lw=1.0, zorder=4)
        j = valid.index('TARC-4') if 'TARC-4' in valid else None
        if j is not None:
            f_min = BASE_FREQ / 4
            ax.hlines(f_min, j - 0.28, j + 0.28,
                      colors='black', linestyles='dashed', lw=0.9, zorder=4)

    ax.set_title(title, fontweight='bold')
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in valid], rotation=30, ha='right', fontsize=7)
    ax.grid(True, axis='y', alpha=0.4, lw=0.5, zorder=0)
    ax.set_axisbelow(True)

    y_min, y_max = ax.get_ylim()
    ax.set_ylim(bottom=min(0, y_min), top=y_max * 1.12)

# ── Legend ─────────────────────────────────────────────────────────────────

present = set(df['Method'].unique())
_legend_spec = [
    ('TARC-4',     dict(facecolor=COLOR_MAP['TARC-4']),                              'TARC (ours)'),
    ('PPO-50Hz',   dict(facecolor=COLOR_MAP['PPO-50Hz']),                            'PPO-50 Hz'),
    ('PPO-20Hz',   dict(facecolor=COLOR_MAP['PPO-20Hz'],   hatch='..', edgecolor=EDGE_MAP['PPO-20Hz']),   'PPO-20 Hz'),
    ('PPO-12.5Hz', dict(facecolor=COLOR_MAP['PPO-12.5Hz'], hatch='//', edgecolor=EDGE_MAP['PPO-12.5Hz']), 'PPO-12.5 Hz'),
]
legend_handles = [mpatches.Patch(**kw, label=lbl) for m, kw, lbl in _legend_spec if m in present]
legend_handles.append(plt.Line2D([0], [0], color='black', lw=0.9, ls='--', label='Min freq (TARC)'))

fig.legend(handles=legend_handles,
           loc='upper center', bbox_to_anchor=(0.5, 1.02),
           ncol=len(legend_handles), frameon=False, fontsize=8)

fig.tight_layout(rect=[0, 0, 1, 0.88])

out_dir = os.path.dirname(os.path.abspath(__file__))
out_pdf = os.path.join(out_dir, 'Go1_Sim_LowFreq_Comparison.pdf')
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_pdf.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
print(f'\nSaved: {out_pdf}')
