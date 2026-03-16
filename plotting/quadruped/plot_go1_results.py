import wandb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

# Use professional style
plt.style.use(['science', 'ieee'])

# Fetch data from arnavsukhija-eth-zurich/Go1_Stability_Rebuttal
api = wandb.Api()
project = "arnavsukhija-eth-zurich/Go1_Stability_Rebuttal"
runs = api.runs(project)

print(f"Fetching data from {project}...")

data = []
EVAL_SEEDS = [42, 43, 44]
SWITCH_COST_VAL = 0.005

for run in runs:
    if run.state != "finished":
        continue
    
    config = run.config
    num_steps = config.get('num_timesteps', 0)
    
    name = run.name
    is_tarc = config.get('switch_cost_wrapper', False)
    base_freq = config.get('control_frequency_hz', 50.0)
    mtr = config.get('max_time_repeat')
    
    summary = {}
    try:
        summary = run.summary._json_dict
    except:
        summary = dict(run.summary)
    
    # Aggregated metrics for this training run (training seed)
    run_rewards = []
    run_actions = []
    run_freqs = []

    for idx in EVAL_SEEDS:
        r_key = f'Results_{idx}/Total reward'
        n_key = f'Results_{idx}/Number of actions'
        f_key = f'Results_{idx}/Avg control frequency (Hz)'
        
        reward = None
        num_actions = None
        freq = None
        
        for k, v in summary.items():
            ks = k.strip()
            if ks == r_key: reward = v
            elif ks == n_key: num_actions = v
            elif ks == f_key: freq = v
            
        if reward is not None:
            run_rewards.append(reward)
            if num_actions is not None:
                run_actions.append(num_actions)
                
                if freq is None:
                    divisor = config.get('base_dt_divisor', 1)
                    e_len = 1000 // divisor
                    c_dt = 0.02 * divisor
                    freq = num_actions / (e_len * c_dt)
                run_freqs.append(freq)

    if run_rewards:
        avg_reward_unpen = np.mean(run_rewards)
        avg_actions = np.mean(run_actions) if run_actions else 0
        avg_pen_reward = avg_reward_unpen - (SWITCH_COST_VAL * avg_actions)
        avg_actual_freq = np.mean(run_freqs) if run_freqs else base_freq

        # Refined Type categorization
        if is_tarc:
            clean_type = f'TARC-{int(mtr)} (600M)'
        else:
            f_str = f"{base_freq:g}" # Removes trailing .0
            if base_freq == 50.0:
                clean_type = f'PPO-50Hz ({num_steps//1_000_000}M)'
            else:
                clean_type = f'PPO-{f_str}Hz (600M)'

        data.append({
            'RunName': name,
            'CleanType': clean_type,
            'Unpenalized Reward': avg_reward_unpen,
            'Penalized Reward': avg_pen_reward,
            'Frequency (Hz)': avg_actual_freq,
            'Steps': num_steps
        })

df = pd.DataFrame(data)

# Define order for plotting
order = [
    'PPO-50Hz (200M)', 
    'PPO-50Hz (600M)', 
    'TARC-4 (600M)', 
    'TARC-5 (600M)', 
    'TARC-10 (600M)', 
    'PPO-12.5Hz (600M)', 
    'PPO-10Hz (600M)', 
    'PPO-5Hz (600M)'
]

# Filtering to only show these specific groups
df = df[df['CleanType'].isin(order)]

# Set up colors
base_colors = sns.color_palette("colorblind")
color_map = {
    'PPO-50Hz (200M)': 'lightgray',
    'PPO-50Hz (600M)': 'gray',
    'TARC-4 (600M)': base_colors[1],
    'TARC-5 (600M)': base_colors[1],
    'TARC-10 (600M)': base_colors[1],
    'PPO-12.5Hz (600M)': base_colors[0],
    'PPO-10Hz (600M)': base_colors[0],
    'PPO-5Hz (600M)': base_colors[0]
}

# --- 2. Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))

metrics = [
    ('Penalized Reward', '(a) Penalized Reward', 'Total Reward'),
    ('Unpenalized Reward', '(b) Unpenalized Reward', 'Total Reward'),
    ('Frequency (Hz)', '(c) Avg. Control Frequency', 'Frequency (Hz)')
]

for i, (metric, title, ylabel) in enumerate(metrics):
    ax = axes[i]
    
    # Group by type and compute Mean and SEM
    group_stats = df.groupby('CleanType')[metric].agg(['mean', 'sem']).reindex(order)
    
    # Filter out empty entries in case some types are missing in the data
    group_stats = group_stats.dropna(how='all')
    valid_order = group_stats.index.tolist()
    
    x = np.arange(len(valid_order))
    means = group_stats['mean'].values
    sems = group_stats['sem'].values
    colors = [color_map.get(t, 'black') for t in valid_order]
    
    # Plot bars with clear error bars
    # Use standard error_kw for prominence
    ax.bar(x, means, yerr=sems, capsize=3, color=colors, width=0.6, error_kw={'lw': 1.0, 'markeredgewidth': 1.0})
    
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_ylabel(ylabel if i == 0 or i == 2 else '', fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(valid_order, rotation=45, ha='right', fontsize=9)
    
    # Set y-axis limits with more headroom for caps
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(bottom=min(0, y_min), top=y_max * 1.25)
    
    # Frequency bounds for (c)
    if metric == 'Frequency (Hz)':
        bound_map = {
            'PPO-50Hz (200M)': 50,
            'PPO-50Hz (600M)': 50,
            'TARC-4 (600M)': 12.5,
            'TARC-5 (600M)': 10,
            'TARC-10 (600M)': 5,
            'PPO-12.5Hz (600M)': 12.5,
            'PPO-10Hz (600M)': 10,
            'PPO-5Hz (600M)': 5
        }
        for j, t in enumerate(valid_order):
            b = bound_map.get(t)
            if b:
                ax.hlines(y=b, xmin=j-0.3, xmax=j+0.3, colors='black', linestyles='dashed', lw=1.0)

plt.tight_layout()
output_filename = 'Go1_Sim_LowFreq_Comparison.pdf'
plt.savefig(output_filename)
plt.savefig(output_filename.replace('.pdf', '.png'), dpi=300)
print(f"Plot saved as {output_filename}")

# Print table for user
print("\n=== Final Aggregate Results ===")
print(group_stats)
