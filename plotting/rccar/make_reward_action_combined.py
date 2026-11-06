"""Combined RC Car hardware-results + actuation-jitter figure for NeurIPS/CoRL.

Generates a single 1×4 figure:
  (a) Penalized reward        — from W&B hardware runs
  (b) Unpenalized reward      — from W&B hardware runs
  (c) Avg. control frequency  — from W&B hardware runs, with min-freq dashes
  (d) Actuation jitter        — Σ|Δthrottle| from trajectory .npy files

Saves RC_Car_HW_Jitter_Combined.{pdf,png} next to this script.

Usage:
    cd plotting/rccar
    python make_reward_action_combined.py [--tarc_only] [--no_latex]
"""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from matplotlib.lines import Line2D
from tueplots import bundles
import wandb

# ── Paths ───────────────────────────────────────────────────────────────────

HERE     = os.path.dirname(os.path.abspath(__file__))
TRAJ_DIR = os.path.join(HERE, '..', '..', 'results', 'rccar', 'trajectories', 'v2', 'origin')

WANDB_PROJECT = 'arnavsukhija-eth-zurich/TaCosFinalResultsv2_origin'
SWITCH_COST   = 0.1

# W&B tag → expected total action count (200-step episode at 30 Hz)
BASE_FREQ = {
    'ppo':               200,
    'hardware_3actions': 100,
    'hardware_4actions':  67,
    'tacos5_hardware':    50,
    'tacos10_hardware':   25,
}

# Trajectory files (relative to TRAJ_DIR) used for jitter computation
PPO_FILES = [
    'PPO/trajectory_hj6j34jo.npy',
    'PPO/trajectory_72pmrwwo.npy',
    'PPO/trajectory_u43ihdue.npy',
]
TARC4_FILES = [
    'Tacos4/trajectory_g9hv0sfq.npy',
    'Tacos4/trajectory_li4evvad.npy',
    'Tacos4/trajectory_ob7yn33s.npy',
]


# ── Trajectory helpers ──────────────────────────────────────────────────────

def _remove_duplicates(traj):
    """Drop consecutive duplicate (state, action, reward, step) rows."""
    out, i = [], 0
    while i < len(traj):
        if (i + 1 < len(traj)
                and np.array_equal(traj[i][0], traj[i+1][0])
                and np.array_equal(traj[i][1], traj[i+1][1])
                and traj[i][2] == traj[i+1][2]
                and traj[i][3] == traj[i+1][3]):
            out.append(traj[i])
            i += 2
        else:
            out.append(traj[i])
            i += 1
    return np.array(out)


def _pseudo_to_steps(pseudo_time: float, max_repeat: int = 4) -> int:
    """Map TARC pseudo_time ∈ [−1, 1] to hold duration in 30-Hz timesteps."""
    dt   = 1.0 / 30
    t_lo = 1 * dt
    t_hi = max_repeat * dt
    t    = (t_hi - t_lo) / 2 * pseudo_time + (t_hi + t_lo) / 2
    return max(1, int(t // dt))


def _jitter_one_file(path: str, is_tarc: bool, max_repeat: int = 4) -> float:
    """Σ|Δthrottle| for a single trajectory file, both agents at 30-Hz resolution."""
    traj    = np.load(path, allow_pickle=True)
    traj    = _remove_duplicates(traj)
    raw_act = list(traj[:, 1])

    if not is_tarc:
        # PPO: terminal action may have 3 elements — clip to 2
        raw_act[-1] = raw_act[-1][:2]

    actions = np.stack([np.asarray(a) for a in raw_act])   # (N, 2 or 3)

    if is_tarc:
        # Exclude terminal outer step, then upsample to 30 Hz
        throttle  = actions[:-1, 1]
        durations = [_pseudo_to_steps(float(actions[k, 2]), max_repeat)
                     for k in range(len(actions) - 1)]
        throttle_30hz = np.repeat(throttle, durations)
        return float(np.abs(np.diff(throttle_30hz)).sum())
    else:
        # PPO already at 30 Hz; exclude terminal step
        throttle = actions[:-1, 1]
        return float(np.abs(np.diff(throttle)).sum())


def _jitter_stats(files, is_tarc: bool, max_repeat: int = 4):
    """Return (mean, SEM) of per-seed jitter over a list of .npy files."""
    vals = [_jitter_one_file(os.path.join(TRAJ_DIR, f), is_tarc, max_repeat)
            for f in files]
    arr  = np.array(vals)
    sem  = arr.std() / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return float(arr.mean()), float(sem)


# ── W&B hardware data ───────────────────────────────────────────────────────

def _get_wandb_data(tags):
    """Fetch (pen_mean, pen_sem, unpen_mean, unpen_sem, freq_mean, freq_sem) per tag."""
    api = wandb.Api()
    pen_m, pen_s, unpen_m, unpen_s, freq_m, freq_s = [], [], [], [], [], []

    def sem(v):
        return float(np.std(v) / np.sqrt(len(v))) if len(v) > 1 else 0.0

    for tag in tags:
        runs    = api.runs(WANDB_PROJECT, {"tags": tag})
        rewards, n_actions = [], []
        for run in runs:
            s = run.summary
            r = s.get("total reward") or s.get("total_reward", 0)
            a = s.get("number of actions", BASE_FREQ[tag])
            if int(a) != BASE_FREQ[tag]:
                a = BASE_FREQ[tag]
            rewards.append(float(r))
            n_actions.append(int(a))

        freqs = [a * 30.0 / 200.0 for a in n_actions]
        pen   = [r - SWITCH_COST * a for r, a in zip(rewards, n_actions)]

        pen_m.append(np.mean(pen));      pen_s.append(sem(pen))
        unpen_m.append(np.mean(rewards)); unpen_s.append(sem(rewards))
        freq_m.append(np.mean(freqs));   freq_s.append(sem(freqs))

    return pen_m, pen_s, unpen_m, unpen_s, freq_m, freq_s


BASE_RATE_HZ = 30.0   # max rate of the RC car
POSTER_FS = 16   # 24 pt does not fit in 8.4 x 2.5 in


def make_poster_figure(names, reward_m, reward_s, jitter_m, jitter_s, out):
    # names: bar labels, TARC first (left), fixed-rate baseline second (right)
    """Two-panel poster figure (8.4 x 2.5 in): task reward | throttle change."""
    plt.rcParams.update({
        'font.size': POSTER_FS, 'axes.labelsize': POSTER_FS, 'xtick.labelsize': POSTER_FS,
        'ytick.labelsize': POSTER_FS, 'axes.linewidth': 1.0,
        'text.usetex': False, 'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'],
        'mathtext.fontset': 'dejavusans',
    })
    pal = sns.color_palette('colorblind')
    colors = [pal[1], '0.55']          # orange = TARC (left), grey = fixed-rate baseline (right)
    x = np.arange(len(names))
    ekw = {'lw': 1.2, 'ecolor': 'black'}

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 2.5))
    for ax, m, sm, ylabel in zip(axes, [reward_m, jitter_m], [reward_s, jitter_s],
                                 ['Task\nreward', 'Throttle\nchange']):
        ax.bar(x, m, 0.55, yerr=sm, color=colors, capsize=4, error_kw=ekw, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis='y', alpha=0.3, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.1)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3, integer=True))
    fig.tight_layout(pad=0.3, w_pad=1.5)
    fig.savefig(out)
    fig.savefig(out.replace('.pdf', '.png'), dpi=300)
    print(f'\nSaved: {out}')


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tarc_only', action='store_true',
                        help='Only Baseline + TARC-4 (default: all variants)')
    parser.add_argument('--poster', action='store_true',
                        help='Two-panel poster figure (task reward + throttle change)')
    parser.add_argument('--no_latex', action='store_true',
                        help='Disable LaTeX rendering (for quick preview)')
    args = parser.parse_args()

    # ── Config ────────────────────────────────────────────────────────────
    if args.tarc_only:
        agent_names = ['Baseline', 'TARC']
        tags        = ['ppo', 'hardware_4actions']
        min_freqs   = [None, 30.0 / 4]
    else:
        agent_names = ['Baseline', 'TARC-4', 'TARC-5', 'TARC-10']
        tags        = ['ppo', 'hardware_4actions', 'tacos5_hardware', 'tacos10_hardware']
        min_freqs   = [None, 30/4, 30/5, 30/10]

    # ── Fetch data ─────────────────────────────────────────────────────────
    print('Fetching W&B hardware data...')
    pen_m, pen_s, unpen_m, unpen_s, freq_m, freq_s = _get_wandb_data(tags)

    print('Computing actuation jitter from trajectory files...')
    ppo_jm,  ppo_js  = _jitter_stats(PPO_FILES,   is_tarc=False)
    tarc4_jm, tarc4_js = _jitter_stats(TARC4_FILES, is_tarc=True, max_repeat=4)

    # Jitter panel always shows Baseline vs TARC(-4)
    j_names  = ['Baseline', 'TARC'] if args.tarc_only else ['Baseline', 'TARC-4']
    j_means  = [ppo_jm,  tarc4_jm]
    j_sems   = [ppo_js,  tarc4_js]

    print(f'  PPO  jitter: {ppo_jm:.2f} ± {ppo_js:.2f}')
    print(f'  TARC jitter: {tarc4_jm:.2f} ± {tarc4_js:.2f}')

    if args.poster:
        # TARC left, baseline right
        names = [f'TARC (ours)\navg $\\approx$ {freq_m[1]:.0f} Hz',
                 f'Fixed rate\n{BASE_RATE_HZ:.0f} Hz']
        make_poster_figure(names, [unpen_m[1], unpen_m[0]], [unpen_s[1], unpen_s[0]],
                           [j_means[1], j_means[0]], [j_sems[1], j_sems[0]],
                           os.path.join(HERE, 'RC_Car_Poster.pdf'))
        return

    # ── Style ──────────────────────────────────────────────────────────────
    rc = bundles.neurips2024()
    if args.no_latex:
        rc.pop('text.latex.preamble', None)
        rc['text.usetex'] = False
    plt.rcParams.update(rc)

    base_colors = sns.color_palette('colorblind')
    n = len(agent_names)
    bar_colors = [base_colors[0]] + [base_colors[1]] * (n - 1)  # blue=baseline, orange=TARC
    j_colors   = [base_colors[0], base_colors[1]]

    # ── Figure ─────────────────────────────────────────────────────────────
    # Match existing hardware-plot width (7.16" = IEEE 2-column text width)
    fig, axes = plt.subplots(1, 4, figsize=(7.16, 1.8))

    bw   = 0.4
    x    = np.arange(n)
    x_j  = np.arange(len(j_names))
    ekw  = {'elinewidth': 0.7, 'markeredgewidth': 0.7, 'ecolor': 'black'}

    # ── Panel (a): Penalized reward ────────────────────────────────────────
    axes[0].bar(x, pen_m, bw, yerr=pen_s, color=bar_colors, capsize=2,
                error_kw=ekw, zorder=3)
    axes[0].set_title('(a) Penalized Reward', fontweight='bold')
    axes[0].set_ylabel('Total Reward')

    # ── Panel (b): Unpenalized reward ──────────────────────────────────────
    axes[1].bar(x, unpen_m, bw, yerr=unpen_s, color=bar_colors, capsize=2,
                error_kw=ekw, zorder=3)
    axes[1].set_title('(b) Unpenalized Reward', fontweight='bold')
    axes[1].set_ylabel('Total Reward')

    # ── Panel (c): Control frequency ───────────────────────────────────────
    axes[2].bar(x, freq_m, bw, yerr=freq_s, color=bar_colors, capsize=2,
                error_kw=ekw, zorder=3)
    axes[2].set_title('(c) Avg. Control Frequency', fontweight='bold')
    axes[2].set_ylabel('Frequency (Hz) $\\downarrow$ better')
    for i, mf in enumerate(min_freqs):
        if mf is not None:
            axes[2].hlines(mf, i - bw/2, i + bw/2,
                           colors='black', lw=0.9, linestyles='dashed', zorder=4)

    # ── Panel (d): Actuation jitter ────────────────────────────────────────
    axes[3].bar(x_j, j_means, bw, yerr=j_sems, color=j_colors, capsize=2,
                error_kw=ekw, zorder=3)
    axes[3].set_title('(d) Motor throttle travel', fontweight='bold')
    if args.no_latex:
        axes[3].set_ylabel('Sum |Delta throttle| $\\downarrow$ better')
    else:
        axes[3].set_ylabel(r'$\sum_t |\Delta\,\mathrm{throttle}_t|$ $\downarrow$ better')

    # ── Shared formatting ──────────────────────────────────────────────────
    for ax, xvals, xnames in zip(axes,
                                  [x, x, x, x_j],
                                  [agent_names, agent_names, agent_names, j_names]):
        ax.set_xticks(xvals)
        ax.set_xticklabels(xnames, rotation=30, ha='right')
        ax.grid(True, axis='y', alpha=0.35, lw=0.5, zorder=0)
        ax.set_axisbelow(True)
        y_lo, y_hi = ax.get_ylim()
        ax.set_ylim(bottom=min(0, y_lo), top=y_hi * 1.15)

    # ── Legend — dashed-line explanation inside the frequency panel ────────
    dash = Line2D([0], [0], color='black', lw=0.9, ls='--', label='Min. freq. (TARC)')
    axes[2].legend(handles=[dash], loc='upper right', frameon=False,
                   fontsize=plt.rcParams.get('legend.fontsize', 7))

    fig.tight_layout()

    # ── Save ───────────────────────────────────────────────────────────────
    out = os.path.join(HERE, 'RC_Car_HW_Jitter_Combined.pdf')
    fig.savefig(out, bbox_inches='tight')
    fig.savefig(out.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
