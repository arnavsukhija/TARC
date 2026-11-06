"""Poster bar chart for the Go1 simulation comparison (6.0 x 3.2 in).

Bars: TARC (avg. control frequency) / fixed 50 Hz / fixed 20 Hz.  y-axis: task reward
(unpenalized).  Expects the per-rollout DataFrame built in plot_go1_results.py
(columns Method, Reward, Freq).
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

POSTER_FS = 18
TARC_AVG_FREQ_HZ = 20   # hardcoded for now (avg. control frequency of TARC)
ORDER = ['TARC-4', 'PPO-50Hz', 'PPO-20Hz']


def make_go1_poster_figure(df, out_pdf):
    plt.rcParams.update({
        'font.size': POSTER_FS, 'axes.labelsize': POSTER_FS, 'xtick.labelsize': POSTER_FS,
        'ytick.labelsize': POSTER_FS, 'axes.linewidth': 1.0,
        'text.usetex': False, 'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'mathtext.fontset': 'dejavusans',
    })
    pal = sns.color_palette('colorblind')
    grey = '0.55'   # fixed-rate baselines are grey on all poster figures
    colors = {'TARC-4': pal[1], 'PPO-50Hz': grey, 'PPO-20Hz': grey}
    hatch = {'TARC-4': '', 'PPO-50Hz': '', 'PPO-20Hz': '//'}

    g = df.groupby('Method')['Reward']
    means, sems = g.mean(), g.sem()
    tarc_freq = TARC_AVG_FREQ_HZ
    labels = {'TARC-4': f'TARC\n(avg. $\\approx${tarc_freq:.0f} Hz)',
              'PPO-50Hz': 'Fixed\n50 Hz', 'PPO-20Hz': 'Fixed\n20 Hz'}

    order = [m for m in ORDER if m in means.index]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    for i, m in enumerate(order):
        ax.bar(i, means[m], width=0.6, yerr=sems[m], color=colors[m], hatch=hatch[m],
               edgecolor='white' if hatch[m] else 'none', linewidth=0,
               capsize=4, error_kw={'lw': 1.2, 'ecolor': 'black'}, zorder=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([labels[m] for m in order])
    ax.set_ylabel('Task reward')
    ax.grid(True, axis='y', alpha=0.3, lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylim(0, means[order].max() * 1.15)
    fig.tight_layout(pad=0.3)
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.replace('.pdf', '.png'), dpi=300)
    print(f'\nSaved: {out_pdf}')
