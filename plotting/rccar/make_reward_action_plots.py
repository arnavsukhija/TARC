import json

import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots


base_control_frequency = {
    'ppo': 200,
    'tacos2_hardware': 200,
    'hardware_3actions': 100,
    'hardware_4actions': 67,
    'tacos5_hardware': 50,
    'tacos10_hardware': 25,
}

# --- 1. Data Processing Function ---
def get_rc_car_data(agent_tags, penalize, switch_cost):
    """
    Fetches and processes RC Car data from Wandb for all specified agents.
    """
    print(f"\n--- Fetching data with penalize={penalize} ---")

    all_reward_means = []
    all_reward_stds = []
    all_freq_means = []
    all_freq_stds = []

    api = wandb.Api()

    for tag in agent_tags:
        runs = api.runs("arnavsukhija-eth-zurich/TaCosFinalResultsv2_origin", {"tags": tag})

        run_rewards = []
        run_actions = []

        for run in runs:
            summary = run.summary
            # Use .get() to avoid errors if keys are missing
            reward = summary.get("total reward", 0)
            reward = summary.get("total_reward", 0) if reward == 0 else reward
            actions = summary.get("number of actions", 0)
            if (actions != base_control_frequency[tag]):
                actions = base_control_frequency[tag]
            # Apply penalty if the flag is set
            if penalize:
                reward -= switch_cost * actions

            run_rewards.append(reward)
            run_actions.append(actions)

        # Convert total actions to average frequency (30Hz base, 200 steps)
        run_freqs = [a * 30.0 / 200.0 for a in run_actions]

        all_reward_means.append(np.mean(run_rewards))
        all_reward_stds.append(np.std(run_rewards) / np.sqrt(len(run_rewards)) if len(run_rewards) > 1 else 0)  # SEM
        all_freq_means.append(np.mean(run_freqs))
        all_freq_stds.append(np.std(run_freqs) / np.sqrt(len(run_freqs)) if len(run_freqs) > 1 else 0)  # SEM

    return all_reward_means, all_reward_stds, all_freq_means, all_freq_stds


# --- 2. Main Plotting Function ---
def create_rc_car_combined_plot(agent_names, pen_reward_means, pen_reward_stds,
                                unpen_reward_means, unpen_reward_stds,
                                freq_means, freq_stds, min_freqs):
    """
    Generates a single, professional 1x3 figure for the RC Car results.
    """
    print("--- Generating combined RC Car results plot... ---")
    plt.style.use(['science', 'ieee'])
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))
    colors = sns.color_palette("colorblind")

    # Plot 1: Penalized Reward


    # Plot 2: Unpenalized Reward
    axes[0].bar(agent_names, unpen_reward_means, yerr=unpen_reward_stds, capsize=3, color=colors, width=0.6)
    axes[0].set_title('Total Reward')

    # Plot 3: Average Control Frequency
    bar_width = 0.6
    axes[1].bar(agent_names, freq_means, width=bar_width, yerr=freq_stds, capsize=3,
                label='Average Frequency', color=colors)
    legend_added = False
    for i, name in enumerate(agent_names):
        if name != 'Baseline':
            # If the legend entry hasn't been added yet, assign the label


            # Set the flag to true so we don't add the label again
            legend_added = True
    axes[1].set_title('Avg. Control Frequency')
    axes[1].set_ylabel('Frequency (Hz)')

    # Global Formatting
    for ax in axes:
        ax.set_xticklabels(agent_names, rotation=45, ha="right")
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(bottom=0, top=y_max * 1.15)

    fig.tight_layout()
    output_filename = 'RC_Car_Combined_Results_video.png'
    plt.savefig(output_filename)
    print(f"--- Final plot saved as: {output_filename} ---")


def create_tarc_sim_vs_hw_plot(agent_names, sim_data, hw_data, min_freqs):
    """
    MODIFIED: Uses a consistent color for 'Sim' and 'Hardware' across all three subplots.
    """
    print("\n--- Generating Sim-vs-Hardware TARC results plot with consistent colors... ---")
    plt.style.use(['science', 'ieee'])
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5))
    colors = sns.color_palette("colorblind")

    # --- NEW: Define consistent colors for Sim and Hardware ---
    sim_color = colors[0]  # Blue for Sim
    hw_color = colors[1]  # Orange for Hardware

    x = np.arange(len(agent_names))
    bar_width = 0.35

    sim_pen_rm, sim_pen_rs, sim_unpen_rm, sim_unpen_rs, sim_fm, sim_fs = sim_data
    hw_pen_rm, hw_pen_rs, hw_unpen_rm, hw_unpen_rs, hw_fm, hw_fs = hw_data

    # Plot 1: Penalized Reward - uses consistent colors
    axes[0].bar(x - bar_width / 2, sim_pen_rm, bar_width, yerr=sim_pen_rs, label='RC Car (Simulation)', capsize=3, color=sim_color)
    axes[0].bar(x + bar_width / 2, hw_pen_rm, bar_width, yerr=hw_pen_rs, label='RC Car (Hardware)', capsize=3, color=hw_color)
    axes[0].set_title('(a) Penalized Reward')
    axes[0].set_ylabel('Total Reward')

    # Plot 2: Unpenalized Reward - uses consistent colors
    axes[1].bar(x - bar_width / 2, sim_unpen_rm, bar_width, yerr=sim_unpen_rs, label='RC Car (Simulation)', capsize=3, color=sim_color)
    axes[1].bar(x + bar_width / 2, hw_unpen_rm, bar_width, yerr=hw_unpen_rs, label='RC Car (Hardware)', capsize=3,
                color=hw_color)
    axes[1].set_title('(b) Unpenalized Reward')

    # Plot 3: Average Control Frequency - uses consistent colors
    axes[2].bar(x - bar_width / 2, sim_fm, bar_width, yerr=sim_fs, label='RC Car (Simulation)', capsize=3, color=sim_color)
    axes[2].bar(x + bar_width / 2, hw_fm, bar_width, yerr=hw_fs, label='RC Car (Hardware)', capsize=3, color=hw_color)
    for i, agent_name in enumerate(agent_names):
        if agent_name != 'Baseline':
            axes[2].hlines(y=min_freqs[i], xmin=i - bar_width, xmax=i + bar_width,
                           colors='black', linestyles='dashed', lw=1.5)
    axes[2].set_title('(c) Avg. Control Frequency')
    axes[2].set_ylabel('Frequency (Hz)')

    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    dashed_line = Line2D([0], [0], color='black', lw=1.5, linestyle='--', label='Frequency Lower Bound')
    handles.append(dashed_line)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, frameon=False)

    # Global Formatting
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(agent_names, rotation=45, ha="right")
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(bottom=0, top=y_max * 1.1)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_filename = 'RC_Car_TARC_SimVsHardware_Results.pdf'
    plt.savefig(output_filename)
    print(f"--- Final plot saved as: {output_filename} ---")

def load_sim_data_from_json(filename, agent_names):
    import os
    if not os.path.exists(filename):
        # fallback to local dir
        filename = os.path.basename(filename)
    try:
        with open(filename, 'r') as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print(f"--- File {filename} not found! Returning Zeroes. ---")
        return [0]*len(agent_names), [0]*len(agent_names), [0]*len(agent_names), [0]*len(agent_names), [0]*len(agent_names), [0]*len(agent_names)

    pen_rewards = all_data.get('penalized_rewards', {})
    unpen_rewards = all_data.get('unpenalized_rewards', {})
    frequencies = all_data.get('frequencies', {})

    sim_pen_rm = [pen_rewards.get(name, {}).get('mean', 0) for name in agent_names]
    sim_pen_rs = [pen_rewards.get(name, {}).get('sem', 0) for name in agent_names]
    sim_unpen_rm = [unpen_rewards.get(name, {}).get('mean', 0) for name in agent_names]
    sim_unpen_rs = [unpen_rewards.get(name, {}).get('sem', 0) for name in agent_names]
    sim_fm = [frequencies.get(name, {}).get('mean', 0) for name in agent_names]
    sim_fs = [frequencies.get(name, {}).get('sem', 0) for name in agent_names]

    return sim_pen_rm, sim_pen_rs, sim_unpen_rm, sim_unpen_rs, sim_fm, sim_fs
    
def create_sim_comparison_plot(agent_names, sim_pen_rm, sim_pen_rs, sim_unpen_rm, sim_unpen_rs, sim_fm, sim_fs, min_freqs):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    print("\\n--- Generating Sim-only comparison plot... ---")
    plt.style.use(['science', 'ieee'])
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.5))

    base_colors = sns.color_palette("colorblind")
    colors = []
    for name in agent_names:
        if 'PPO-30' in name or name == 'Baseline':
            colors.append('gray')
        elif 'PPO' in name:
            colors.append(base_colors[0])
        elif 'TARC' in name:
            colors.append(base_colors[1])
        else:
            colors.append('gray')

    x = np.arange(len(agent_names))
    bar_width = 0.6

    # Plot 1: Penalized Reward
    axes[0].bar(x, sim_pen_rm, bar_width, yerr=sim_pen_rs, capsize=3, color=colors)
    axes[0].set_title('(a) Penalized Reward')
    axes[0].set_ylabel('Total Reward')

    # Plot 2: Unpenalized Reward
    axes[1].bar(x, sim_unpen_rm, bar_width, yerr=sim_unpen_rs, capsize=3, color=colors)
    axes[1].set_title('(b) Unpenalized Reward')

    # Plot 3: Average Control Frequency
    axes[2].bar(x, sim_fm, bar_width, yerr=sim_fs, capsize=3, color=colors)
    for i, agent_name in enumerate(agent_names):
        if min_freqs and i < len(min_freqs) and min_freqs[i] is not None:
            axes[2].hlines(y=min_freqs[i], xmin=i - bar_width/2, xmax=i + bar_width/2,
                           colors='black', linestyles='dashed', lw=1.5)
    axes[2].set_title('(c) Avg. Control Frequency')
    axes[2].set_ylabel('Frequency (Hz)')

    # Global Formatting
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(agent_names, rotation=45, ha="right", fontsize=9)
        y_min, y_max = ax.get_ylim()
        ax.set_ylim(bottom=min(0, y_min), top=y_max * 1.15)

    custom_lines = [
        Patch(facecolor='gray', label='High-Freq Baseline (Fixed)'),
        Patch(facecolor=base_colors[0], label='Low-Freq Baseline (Fixed)'),
        Patch(facecolor=base_colors[1], label='TARC (Adaptive)'),
        Line2D([0], [0], color='black', lw=1.5, linestyle='--', label='Min Freq Bound')
    ]
    fig.legend(handles=custom_lines, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, frameon=False, fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.9])

    output_filename = 'RC_Car_Sim_LowFreq_Comparison.pdf'
    plt.savefig(output_filename)
    print(f"--- Final plot saved as: {output_filename} ---")

# --- 3. Main Execution Block ---
if __name__ == '__main__':
    agent_names = ['Baseline', 'TARC-4']  # We want TARC-4 here
    # TARC-4 was max_time_repeat=3, which is 10Hz.
    
    wandb_tags = ['ppo', 'hardware_4actions']
    minimum_frequencies = [30, 7.5]

    mean_pen_rewards, std_pen_rewards, _, _ = get_rc_car_data(wandb_tags, penalize=True, switch_cost=0.1)
    mean_unpen_rewards, std_unpen_rewards, mean_frequencies, std_frequencies = get_rc_car_data(wandb_tags, penalize=False, switch_cost=0.1)
    hw_data = (mean_pen_rewards, std_pen_rewards, mean_unpen_rewards, std_unpen_rewards, mean_frequencies, std_frequencies)

    # Make sure we read from the absolute path or locally
    import os
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'RCCar_simulation_results.json')
    sim_data = load_sim_data_from_json(json_path, agent_names)
    create_tarc_sim_vs_hw_plot(agent_names, sim_data, hw_data, minimum_frequencies)
    create_rc_car_combined_plot(
        agent_names=['Baseline', 'TARC'], # HW names
        pen_reward_means=mean_pen_rewards,
        pen_reward_stds=std_pen_rewards,
        unpen_reward_means=mean_unpen_rewards,
        unpen_reward_stds=std_unpen_rewards,
        freq_means=mean_frequencies,
        freq_stds=std_frequencies,
        min_freqs=minimum_frequencies
    )
    
    # === SIM-ONLY COMPARISON PLOT ===
    # Compare Baseline, PPO (Low Freq), and TARC variants matching frequency bounds
    target_agents = [
        'Baseline', 
        'PPO-15', 
        'PPO-10', 'TARC-3', 
        'PPO-7.5', 'TARC-4', 
        'TARC-5', 
        'TARC-10'
    ]
    
    comp_agents = [
        'PPO-30', 
        'PPO-15', 
        'PPO-10', 'TARC-3', 
        'PPO-7.5', 'TARC-4', 
        'TARC-5', 
        'TARC-10'
    ]
    sim_data_comp = load_sim_data_from_json(json_path, target_agents)
    
    comp_pen_rm, comp_pen_rs, comp_unpen_rm, comp_unpen_rs, comp_fm, comp_fs = sim_data_comp
    
    comp_min_freqs = [30, 15, 10, 10, 7.5, 7.5, 6, 3]
    
    create_sim_comparison_plot(
        comp_agents,
        comp_pen_rm, comp_pen_rs,
        comp_unpen_rm, comp_unpen_rs,
        comp_fm, comp_fs,
        comp_min_freqs
    )