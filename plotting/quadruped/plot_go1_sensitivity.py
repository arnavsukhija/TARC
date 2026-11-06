import os
import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

# Use professional style
plt.style.use(['science', 'ieee'])

# WandB Setup
api = wandb.Api()
project = "arnavsukhija-eth-zurich/Go1_Sensitivity_Rebuttal"
runs = api.runs(project)

print(f"Fetching data from {project}...")

data = []
EVAL_SEEDS = [42, 43, 44]

# TARC Theoretical Lower Bounds
MIN_FREQS = {3: 16.67, 4: 12.5, 5: 10.0, 10: 5.0}

for run in runs:
    if run.state != "finished":
        continue
    
    config = run.config
    sc = config.get('switch_cost')
    mtr = config.get('max_time_repeat')
    
    if sc is None or mtr is None:
        continue

    summary = {}
    try:
        summary = run.summary._json_dict
    except:
        summary = dict(run.summary)
    
    # Aggregated metrics for this training run (training seed)
    run_rewards = []
    run_freqs = []

    for idx in EVAL_SEEDS:
        r_key = f'Results_{idx}/Total reward'
        n_key = f'Results_{idx}/Number of actions'
        f_key = f'Results_{idx}/Avg control frequency (Hz)'
        
        reward = None
        num_actions = None
        freq = None
        
        # Robust key handling (strip spaces)
        for k, v in summary.items():
            ks = k.strip()
            if ks == r_key: reward = v
            elif ks == n_key: num_actions = v
            elif ks == f_key: freq = v
            
        if reward is not None:
            run_rewards.append(reward)
            if num_actions is not None:
                # If freq not logged, calculate it
                if freq is None:
                    divisor = config.get('base_dt_divisor', 1)
                    e_len = 1000 // divisor
                    c_dt = 0.02 * divisor
                    freq = num_actions / (e_len * c_dt)
                run_freqs.append(freq)

    if run_rewards:
        # Compute means over the 3 eval seeds for this specific training run
        avg_reward_unpen = np.mean(run_rewards)
        avg_actual_freq = np.mean(run_freqs) if run_freqs else (50.0 / mtr)

        data.append({
            'RunName': run.name,
            'SwitchCost': sc,
            'MTR': mtr,
            'Type': f'TARC-{mtr}',
            'Unpenalized Reward': avg_reward_unpen,
            'Frequency (Hz)': avg_actual_freq
        })

df = pd.DataFrame(data)

# Sort switch costs for clean plotting
sc_order = sorted(df['SwitchCost'].unique())

# Setup colors
base_colors = sns.color_palette("colorblind")
mtr_types = sorted(df['MTR'].unique())
color_map = {mtr: base_colors[i % len(base_colors)] for i, mtr in enumerate(mtr_types)}

# --- Plotting ---
fig, axes = plt.subplots(1, 2, figsize=(7, 3))

# Plot 1: Unpenalized Reward
for mtr in mtr_types:
    subset = df[df['MTR'] == mtr]
    # Group by switch cost to get mean and sem
    stats = subset.groupby('SwitchCost')['Unpenalized Reward'].agg(['mean', 'sem']).reset_index()
    stats = stats.sort_values('SwitchCost')
    
    axes[0].plot(stats['SwitchCost'], stats['mean'], marker='o', label=f'TARC-{mtr}', color=color_map[mtr], markersize=3)
    axes[0].fill_between(stats['SwitchCost'], 
                         stats['mean'] - stats['sem'], 
                         stats['mean'] + stats['sem'], 
                         alpha=0.2, color=color_map[mtr])

axes[0].set_xscale('log')
axes[0].set_title('(a) Unpenalized Reward')
axes[0].set_ylabel('Total Reward')
axes[0].set_xlabel('Switch Cost $c$')
axes[0].set_xticks(sc_order)
axes[0].set_xticklabels([f'{x:g}' for x in sc_order], rotation=30)
axes[0].minorticks_off()

# Plot 2: Control Frequency
for mtr in mtr_types:
    subset = df[df['MTR'] == mtr]
    stats = subset.groupby('SwitchCost')['Frequency (Hz)'].agg(['mean', 'sem']).reset_index()
    stats = stats.sort_values('SwitchCost')
    
    axes[1].plot(stats['SwitchCost'], stats['mean'], marker='o', label=f'TARC-{mtr}', color=color_map[mtr], markersize=3)
    axes[1].fill_between(stats['SwitchCost'], 
                         stats['mean'] - stats['sem'], 
                         stats['mean'] + stats['sem'], 
                         alpha=0.2, color=color_map[mtr])
    
    # Draw theoretical min bound
    min_f = MIN_FREQS.get(mtr, 50.0/mtr)
    axes[1].axhline(y=min_f, color=color_map[mtr], linestyle='--', lw=0.8, alpha=0.6)

axes[1].set_xscale('log')
axes[1].set_title('(b) Avg. Control Frequency')
axes[1].set_ylabel('Frequency (Hz)')
axes[1].set_xlabel('Switch Cost $c$')
axes[1].set_xticks(sc_order)
axes[1].set_xticklabels([f'{x:g}' for x in sc_order], rotation=30)
axes[1].minorticks_off()

# Legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(mtr_types), frameon=False)

plt.tight_layout()
output_filename = 'Go1_SwitchCost_Sensitivity.png'
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
plt.savefig(output_filename.replace('.png', '.pdf'), bbox_inches='tight')
print(f"Plot saved as {output_filename}")

# Print summary
print("\n=== Aggregate Results ===")
print(df.groupby(['MTR', 'SwitchCost'])[['Unpenalized Reward', 'Frequency (Hz)']].mean())
