import os
import wandb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
from collections import defaultdict

# Maps max_time_repeat -> TARC agent name
TARC_NAMES = {
    3: 'TARC-3',
    4: 'TARC-4',
    5: 'TARC-5',
    10: 'TARC-10',
}

# The physical minimum frequency = 30 / max_time_repeat
MIN_FREQS = {3: 10.0, 4: 7.5, 5: 6.0, 10: 3.0}

SWITCH_COSTS = [0.01, 0.05, 0.1, 0.2, 0.5]
N_EVAL = 9  # results/total_reward_0 ... results/total_reward_8


def fetch_sweep_data():
    api = wandb.Api()
    runs = api.runs("arnavsukhija-eth-zurich/TARC_RCCar_SwitchCosts")

    # results[(switch_cost, max_time_repeat)] = list of (mean_unpen_reward, mean_pen_reward, mean_freq) per seed
    results = defaultdict(list)

    for run in runs:
        c = run.config
        s = run.summary
        sc = c.get("switch_cost")
        mtr = c.get("max_time_repeat")
        # episode_time is the physical duration of the episode in seconds (from config)
        ep_time = c.get("episode_time", 200.0 / 30.0)

        unpen_rewards, actions = [], []
        for i in range(N_EVAL):
            r = s.get(f"results/total_reward_{i}")
            a = s.get(f"results/num_actions_{i}")
            if r is not None:
                unpen_rewards.append(r)  # eval uses switch_cost=0, so this is unpenalized
            if a is not None:
                actions.append(a)

        if not unpen_rewards:
            continue

        mean_unpen = np.mean(unpen_rewards)
        mean_actions = np.mean(actions) if actions else 0
        # Penalized reward = unpenalized - switch_cost * num_actions
        mean_pen = mean_unpen - sc * mean_actions
        # Correct frequency: macro-steps per second = num_actions / episode_time
        mean_freq = mean_actions / ep_time

        results[(sc, mtr)].append((mean_unpen, mean_pen, mean_freq))

    return results


def create_sensitivity_plot(results):
    plt.style.use(['science', 'ieee'])
    base_colors = sns.color_palette("colorblind")

    max_time_repeats = sorted(TARC_NAMES.keys())  # [3, 4, 5, 10]
    colors = {mtr: base_colors[i] for i, mtr in enumerate(max_time_repeats)}

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    for mtr in max_time_repeats:
        name = TARC_NAMES[mtr]
        color = colors[mtr]
        min_freq = MIN_FREQS[mtr]

        sc_vals, unpen_means, unpen_sems = [], [], []
        freq_means, freq_sems = [], []

        for sc in SWITCH_COSTS:
            seed_data = results.get((sc, mtr), [])
            if not seed_data:
                continue
            u = [d[0] for d in seed_data]
            f = [d[2] for d in seed_data]
            n = len(u)

            sc_vals.append(sc)
            unpen_means.append(np.mean(u))
            unpen_sems.append(np.std(u) / np.sqrt(n) if n > 1 else 0)
            freq_means.append(np.mean(f))
            freq_sems.append(np.std(f) / np.sqrt(n) if n > 1 else 0)

        sc_arr = np.array(sc_vals)

        # Plot 1: Unpenalized Reward
        axes[0].plot(sc_arr, unpen_means, marker='o', label=name, color=color)
        axes[0].fill_between(sc_arr,
                             np.array(unpen_means) - np.array(unpen_sems),
                             np.array(unpen_means) + np.array(unpen_sems),
                             alpha=0.2, color=color)

        # Plot 2: Avg Control Frequency
        axes[1].plot(sc_arr, freq_means, marker='o', label=name, color=color)
        axes[1].fill_between(sc_arr,
                             np.array(freq_means) - np.array(freq_sems),
                             np.array(freq_means) + np.array(freq_sems),
                             alpha=0.2, color=color)
        # Min frequency bound (dashed)
        axes[1].axhline(y=min_freq, color=color, linestyle='--', lw=1.0, alpha=0.7)

    axes[0].set_title('(a) Unpenalized Reward')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_xlabel('Switch Cost')

    axes[1].set_title('(b) Avg. Control Frequency')
    axes[1].set_ylabel('Frequency (Hz)')
    axes[1].set_xlabel('Switch Cost')

    for ax in axes:
        ax.set_xscale('log')
        ax.set_xticks(SWITCH_COSTS)
        ax.set_xticklabels([str(sc) for sc in SWITCH_COSTS], rotation=30)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels,
               loc='upper center', bbox_to_anchor=(0.5, 1.08),
               ncol=len(max_time_repeats), frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.92])
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RC_Car_SwitchCost_Sensitivity.pdf')
    plt.savefig(output_path)
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300)
    print(f"--- Plot saved: {output_path} ---")
    return output_path


if __name__ == '__main__':
    print("Fetching data from WandB...")
    results = fetch_sweep_data()
    print(f"Groups loaded: {list(results.keys())}")
    create_sensitivity_plot(results)
