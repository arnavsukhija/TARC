"""
Combined switch-cost sensitivity figure: RC Car (left) | Go1 (right).
Saves Combined_SwitchCost_Sensitivity.{pdf,png} next to this script.
"""
import os
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import scienceplots
from collections import defaultdict
from matplotlib.lines import Line2D

plt.style.use(['science', 'ieee'])

# ── Shared constants ───────────────────────────────────────────────────────
EVAL_SEEDS   = [42, 43, 44]
TARC_NAMES   = {3: 'TARC-3', 4: 'TARC-4', 5: 'TARC-5', 10: 'TARC-10'}
MTRS         = sorted(TARC_NAMES.keys())
base_colors  = sns.color_palette("colorblind")
COLORS       = {mtr: base_colors[i] for i, mtr in enumerate(MTRS)}

RC_SWITCH_COSTS = [0.01, 0.05, 0.1, 0.2, 0.5]
RC_MIN_FREQS    = {3: 10.0, 4: 7.5, 5: 6.0, 10: 3.0}   # 30 / N
GO1_MIN_FREQS   = {3: 50/3, 4: 12.5, 5: 10.0, 10: 5.0}  # 50 / N


# ── Data fetchers ──────────────────────────────────────────────────────────

def fetch_rccar_data():
    api  = wandb.Api()
    runs = api.runs("arnavsukhija-eth-zurich/TARC_RCCar_SwitchCosts")
    results = defaultdict(list)
    for run in runs:
        c   = run.config
        s   = run.summary
        sc  = c.get("switch_cost")
        mtr = c.get("max_time_repeat")
        if sc is None or mtr is None:
            continue
        ep_time = c.get("episode_time", 200.0 / 30.0)
        unpen_rewards, actions = [], []
        for i in range(9):
            r = s.get(f"results/total_reward_{i}")
            a = s.get(f"results/num_actions_{i}")
            if r is not None: unpen_rewards.append(r)
            if a is not None: actions.append(a)
        if not unpen_rewards:
            continue
        mean_freq = np.mean(actions) / ep_time if actions else 0.0
        results[(sc, mtr)].append((np.mean(unpen_rewards), mean_freq))
    return results


def fetch_go1_data():
    api  = wandb.Api()
    runs = api.runs("arnavsukhija-eth-zurich/Go1_Sensitivity_Rebuttal")
    rows = []
    for run in runs:
        if run.state != "finished":
            continue
        config = run.config
        sc  = config.get('switch_cost')
        mtr = config.get('max_time_repeat')
        if sc is None or mtr is None:
            continue
        try:
            summary = run.summary._json_dict
        except Exception:
            summary = dict(run.summary)
        run_rewards, run_freqs = [], []
        for idx in EVAL_SEEDS:
            r_key = f'Results_{idx}/Total reward'
            n_key = f'Results_{idx}/Number of actions'
            f_key = f'Results_{idx}/Avg control frequency (Hz)'
            reward = num_actions = freq = None
            for k, v in summary.items():
                ks = k.strip()
                if ks == r_key:   reward = v
                elif ks == n_key: num_actions = v
                elif ks == f_key: freq = v
            if reward is not None:
                run_rewards.append(reward)
                if num_actions is not None:
                    if freq is None:
                        divisor = config.get('base_dt_divisor', 1)
                        e_len   = 1000 // divisor
                        c_dt    = 0.02 * divisor
                        freq    = num_actions / (e_len * c_dt)
                    run_freqs.append(freq)
        if run_rewards:
            rows.append({
                'SwitchCost':        sc,
                'MTR':               mtr,
                'Unpenalized Reward': np.mean(run_rewards),
                'Frequency (Hz)':    np.mean(run_freqs) if run_freqs else (50.0 / mtr),
            })
    return pd.DataFrame(rows)


# ── Plotting ───────────────────────────────────────────────────────────────

def _fmt_xaxis(ax, sc_list):
    ax.set_xscale('log')
    ax.set_xticks(sc_list)
    ax.set_xticklabels([f'{x:g}' for x in sc_list], rotation=30, ha='right')
    ax.minorticks_off()


def plot_combined(rc_results, go1_df, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.8))

    go1_sc_list = sorted(go1_df['SwitchCost'].unique()) if not go1_df.empty else RC_SWITCH_COSTS

    for mtr in MTRS:
        name  = TARC_NAMES[mtr]
        color = COLORS[mtr]

        # ── RC Car ────────────────────────────────────────────────────────
        sc_vals, rew_m, rew_s, frq_m, frq_s = [], [], [], [], []
        for sc in RC_SWITCH_COSTS:
            pts = rc_results.get((sc, mtr), [])
            if not pts:
                continue
            u = [d[0] for d in pts]
            f = [d[1] for d in pts]
            n = len(u)
            sc_vals.append(sc)
            rew_m.append(np.mean(u));  rew_s.append(np.std(u) / np.sqrt(n) if n > 1 else 0)
            frq_m.append(np.mean(f));  frq_s.append(np.std(f) / np.sqrt(n) if n > 1 else 0)

        if sc_vals:
            sa = np.array(sc_vals)
            rm, rs, fm, fs = map(np.array, (rew_m, rew_s, frq_m, frq_s))
            axes[0, 0].plot(sa, rm, marker='o', ms=3, label=name, color=color)
            axes[0, 0].fill_between(sa, rm - rs, rm + rs, alpha=0.15, color=color)
            axes[1, 0].plot(sa, fm, marker='o', ms=3, label=name, color=color)
            axes[1, 0].fill_between(sa, fm - fs, fm + fs, alpha=0.15, color=color)
            axes[1, 0].axhline(RC_MIN_FREQS[mtr], color=color, ls='--', lw=0.8, alpha=0.6)

        # ── Go1 ───────────────────────────────────────────────────────────
        sub = go1_df[go1_df['MTR'] == mtr]
        if sub.empty:
            continue
        sr = sub.groupby('SwitchCost')['Unpenalized Reward'].agg(['mean', 'sem']).reset_index().sort_values('SwitchCost')
        sf = sub.groupby('SwitchCost')['Frequency (Hz)'].agg(['mean', 'sem']).reset_index().sort_values('SwitchCost')

        axes[0, 1].plot(sr['SwitchCost'], sr['mean'], marker='o', ms=3, label=name, color=color)
        axes[0, 1].fill_between(sr['SwitchCost'], sr['mean'] - sr['sem'], sr['mean'] + sr['sem'], alpha=0.15, color=color)
        axes[1, 1].plot(sf['SwitchCost'], sf['mean'], marker='o', ms=3, label=name, color=color)
        axes[1, 1].fill_between(sf['SwitchCost'], sf['mean'] - sf['sem'], sf['mean'] + sf['sem'], alpha=0.15, color=color)
        axes[1, 1].axhline(GO1_MIN_FREQS[mtr], color=color, ls='--', lw=0.8, alpha=0.6)

    # ── Titles and labels ─────────────────────────────────────────────────
    axes[0, 0].set_title('(a) RC Car — Unpenalized Reward')
    axes[0, 1].set_title('(b) Go1 — Unpenalized Reward')
    axes[1, 0].set_title('(c) RC Car — Avg. Control Frequency')
    axes[1, 1].set_title('(d) Go1 — Avg. Control Frequency')

    axes[0, 0].set_ylabel('Total Reward')
    axes[1, 0].set_ylabel('Frequency (Hz)')
    axes[1, 0].set_xlabel('Switch Cost $c$')
    axes[1, 1].set_xlabel('Switch Cost $c$')

    _fmt_xaxis(axes[0, 0], RC_SWITCH_COSTS)
    _fmt_xaxis(axes[1, 0], RC_SWITCH_COSTS)
    _fmt_xaxis(axes[0, 1], go1_sc_list)
    _fmt_xaxis(axes[1, 1], go1_sc_list)

    # Remove x-tick labels from top row to reduce clutter
    for ax in axes[0]:
        ax.set_xticklabels([])
        ax.set_xlabel('')

    # ── Shared legend ─────────────────────────────────────────────────────
    handles, labels = axes[0, 0].get_legend_handles_labels()
    dashed = Line2D([0], [0], color='gray', lw=0.8, ls='--', label='Min Freq Bound')
    handles.append(dashed)
    labels.append('Min Freq Bound')
    fig.legend(handles=handles, labels=labels,
               loc='upper center', bbox_to_anchor=(0.5, 1.04),
               ncol=len(MTRS) + 1, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_pdf = os.path.join(output_dir, 'Combined_SwitchCost_Sensitivity.pdf')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.savefig(out_pdf.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {out_pdf}")


if __name__ == '__main__':
    print("Fetching RC Car sensitivity data...")
    rc_results = fetch_rccar_data()
    print(f"  RC Car groups: {len(rc_results)}")

    print("Fetching Go1 sensitivity data...")
    go1_df = fetch_go1_data()
    print(f"  Go1 runs: {len(go1_df)}")

    out_dir = os.path.dirname(os.path.abspath(__file__))
    plot_combined(rc_results, go1_df, out_dir)
