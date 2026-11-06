import pandas as pd
from tueplots import figsizes, bundles

from wtc.envs.rccar import decode_angles_numpy, rotate_coordinates
import wandb
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.stats import sem
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
def correct_all_duplicate_pairs(trajectories):
    corrected_trajectories = []
    i = 0
    while i < len(trajectories):
        if i + 1 < len(trajectories) and np.array_equal(trajectories[i][0], trajectories[i + 1][0]) and np.array_equal(trajectories[i][1], trajectories[i+1][1]) and trajectories[i][2]==trajectories[i+1][2] and trajectories[i][3]==trajectories[i+1][3]:
            print(f"Detected and removed duplicate trajectory at index {i+1}.")
            corrected_trajectories.append(trajectories[i])  # Keep one of the duplicates
            i += 2  # Skip both the current and the next (duplicate)
        else:
            corrected_trajectories.append(trajectories[i])
            i += 1
    return np.array(corrected_trajectories)

def compute_time(pseudo_time: float,
                 ) -> int:
    dt = (1.0 / 30)
    t_upper = 4 * dt
    t_lower = 1 * dt
    time_for_action = ((t_upper - t_lower) / 2 * pseudo_time + (t_upper + t_lower) / 2) #pseudo time for action is between [-1,1], we map it to tmin, tmax
    return (time_for_action // dt)

def plot_state_trajectory(state_list, step_counts, title="State Trajectory", scale_factor=2.0):
    state_array = np.array(state_list)
    num_components = state_array.shape[1]

    labels = ['x', 'y', 'theta', 'velocity x', 'velocity y', 'angular velocity']
    if len(labels) < num_components:
        labels = [f"Component {i}" for i in range(num_components)]

    # Create a larger figure with adjusted base dimensions
    fig, axes = plt.subplots(nrows=2, ncols=3,
                             figsize=(scale_factor * 15, scale_factor * 10))  # Increased base size
    axes = axes.flatten()

    for i in range(num_components):
        axes[i].plot(step_counts, state_array[:, i], label=labels[i], color="b", linewidth=2)
        axes[i].set_xlabel("Step", fontsize=14)
        axes[i].set_ylabel(labels[i], fontsize=14)
        axes[i].set_title(labels[i], fontsize=16)
        for x in step:
            axes[i].axvline(x=x, color='gray', linestyle='--', linewidth=0.7)

        # Format ticks
        axes[i].tick_params(axis='x', which='major', labelsize=12)
        axes[i].tick_params(axis='y', which='major', labelsize=12)
        axes[i].xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))  # Ensures 5 x-ticks
        axes[i].yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))


        # Styling
        axes[i].legend(fontsize=12)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].spines['bottom'].set_linewidth(1)
        axes[i].spines['left'].set_linewidth(1)

    # Remove empty subplots
    for i in range(num_components, len(axes)):
        fig.delaxes(axes[i])

    # Add main title and adjust spacing
    fig.suptitle(title, fontsize=18, fontweight="bold")
    plt.subplots_adjust(hspace=1.5, wspace=0.6)  # Increased spacing between subplots
    plt.show()

def plot_rewards(rewards, steps, title="Running Rewards", scale_factor=2.0):
    plt.figure(figsize=(scale_factor * 10, scale_factor * 6))
    plt.plot(steps, rewards, label="Reward", color="g", linewidth=2)
    plt.xlabel("Step Count", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.title(title, fontsize=16)
    plt.tick_params(axis='x', which='major', labelsize=12)
    plt.tick_params(axis='y', which='major', labelsize=12)
    plt.legend(fontsize=12)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))

    for x_val in step:
        ax.axvline(x=x_val, color='gray', linestyle='--', linewidth=0.7, zorder=2)

    plt.tight_layout()
    plt.show()

OUTPUT_PLOT_FILENAME = 'RCCar_action_deltas.pdf'


def plot_averaged_action_deltas(tarc_actions_seeds, tarc_steps_seeds,
                                baseline_actions_seeds, baseline_steps_seeds,
                                title="Average Action Smoothness"):
    """
    Processes multiple seeds for TARC and baseline policies and plots the
    averaged total sum of action deltas with standard error bars.
    """

    tarc_total_deltas = []
    baseline_total_deltas = []

    # --- Process TARC Data for each seed ---
    for actions, steps in zip(tarc_actions_seeds, tarc_steps_seeds):
        df_tarc = pd.DataFrame(actions, columns=['steering', 'throttle', 'action_duration'])
        df_tarc['action_duration'] = df_tarc['action_duration'].apply(compute_time)
        df_tarc = df_tarc.iloc[:-1]
        df_tarc_upsampled = df_tarc.loc[df_tarc.index.repeat(df_tarc['action_duration'])]
        total_delta = np.linalg.norm(df_tarc_upsampled[['throttle']].diff().dropna().values, axis=1).sum()
        tarc_total_deltas.append(total_delta)

    # --- Process Baseline Data for each seed ---
    for actions, steps in zip(baseline_actions_seeds, baseline_steps_seeds):
        df_baseline = pd.DataFrame(actions, columns=['steering', 'throttle'])
        df_baseline = df_baseline.iloc[:-1]
        total_delta = np.linalg.norm(df_baseline[['throttle']].diff().dropna().values, axis=1).sum()
        baseline_total_deltas.append(total_delta)

    # --- Calculate Mean and Standard Error of the Mean (SEM) ---
    baseline_mean = np.mean(baseline_total_deltas)
    baseline_sem = np.std(baseline_total_deltas) / np.sqrt(len(baseline_total_deltas))

    tarc_mean = np.mean(tarc_total_deltas)
    tarc_sem = np.std(tarc_total_deltas) / np.sqrt(len(tarc_total_deltas))

    print(f"--- Averaged Total Action Deltas ---")
    print(f"Baseline Policy: Mean = {baseline_mean:.2f}, SEM = {baseline_sem:.2f}")
    print(f"TARC Policy:     Mean = {tarc_mean:.2f}, SEM = {tarc_sem:.2f}")

    # --- Plotting ---
    plt.style.use(['science', 'ieee'])
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    labels = ['Baseline Policy', 'TARC-4 Policy']
    means = [baseline_mean, tarc_mean]
    sems = [baseline_sem, tarc_sem]
    colors = ['lightcoral', 'lightblue']

    bars = ax.bar(labels, means, yerr=sems, color=colors, capsize=4)
    ax.bar_label(bars, fmt='%.2f', padding=3)

    ax.set_ylabel('Jitter $||\\mathbf{a}_t - \\mathbf{a}_{t-1}||$', fontsize=12)
    ax.set_ylim(bottom=0, top=max(means) * 1.2)
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT_FILENAME, dpi=300)
    print(f"\nPlot saved successfully to '{OUTPUT_PLOT_FILENAME}'")
    plt.show()

# PPO Policies
# seed 1
file_path = "../../results/rccar/trajectories/v2/origin/PPO/trajectory_hj6j34jo.npy"
trajectory = np.load(file_path, allow_pickle=True)
trajectory = correct_all_duplicate_pairs(trajectory)
state = trajectory[:, 0]
state = np.stack(state)
action = np.array(trajectory[:, 1])
action[200] = action[200][:2] #clip the last action
action = np.stack(action)
reward = trajectory[:, 2]
step = trajectory[:, 3]
ppo_steps_seed1 = step
ppo_actions_seed1 = action

# seed 0
file_path = "../../results/rccar/trajectories/v2/origin/PPO/trajectory_72pmrwwo.npy"
trajectory = np.load(file_path, allow_pickle=True)
trajectory = correct_all_duplicate_pairs(trajectory)
state = trajectory[:, 0]
state = np.stack(state)
action = np.array(trajectory[:, 1])
action[200] = action[200][:2] #clip the last action
action = np.stack(action)
reward = trajectory[:, 2]
step = trajectory[:, 3]
ppo_steps_seed0 = step
ppo_actions_seed0 = action

# seed 2
file_path = "../../results/rccar/trajectories/v2/origin/PPO/trajectory_u43ihdue.npy"
trajectory = np.load(file_path, allow_pickle=True)
trajectory = correct_all_duplicate_pairs(trajectory)
state = trajectory[:, 0]
state = np.stack(state)
action = np.array(trajectory[:, 1])
action[200] = action[200][:2] #clip the last action
action = np.stack(action)
reward = trajectory[:, 2]
step = trajectory[:, 3]
ppo_steps_seed2 = step
ppo_actions_seed2 = action


## TARC-3 policies
# seed 0
file_path = "../../results/rccar/trajectories/v2/origin/TARC4/trajectory_g9hv0sfq.npy"
trajectory = np.load(file_path, allow_pickle=True)
trajectory = correct_all_duplicate_pairs(trajectory)
state = trajectory[:, 0]
state = np.stack(state)
action = np.array(trajectory[:, 1])
action = np.stack(action)
reward = trajectory[:, 2]
step = trajectory[:, 3]
tarc_step_seed0 = step
tarc_action_seed0 = action

# seed 2
file_path = "../../results/rccar/trajectories/v2/origin/TARC4/trajectory_li4evvad.npy"
trajectory = np.load(file_path, allow_pickle=True)
trajectory = correct_all_duplicate_pairs(trajectory)
state = trajectory[:, 0]
state = np.stack(state)
action = np.array(trajectory[:, 1])
action = np.stack(action)
reward = trajectory[:, 2]
step = trajectory[:, 3]
tarc_step_seed2 = step
tarc_action_seed2 = action

file_path = "../../results/rccar/trajectories/v2/origin/TARC4/trajectory_ob7yn33s.npy"
trajectory = np.load(file_path, allow_pickle=True)
trajectory = correct_all_duplicate_pairs(trajectory)
state = trajectory[:, 0]
state = np.stack(state)
action = np.array(trajectory[:, 1])
action = np.stack(action)
reward = trajectory[:, 2]
step = trajectory[:, 3]
tarc_step_seed1 = step
tarc_action_seed1 = action



plot_averaged_action_deltas([tarc_action_seed0, tarc_action_seed1, tarc_action_seed2], [tarc_step_seed0, tarc_step_seed1, tarc_step_seed2]
                            , [ppo_actions_seed0, ppo_actions_seed1, ppo_actions_seed2], [ppo_steps_seed0, ppo_steps_seed1, ppo_steps_seed2], 'Throttle Deltas over episode')
exit()
def plot_cumulative_action_cost(tarc_actions, tarc_steps, baseline_actions, baseline_steps,
                                title="Cumulative Control Effort"):
    """
    Processes TARC and baseline actions and plots the cumulative sum of squared components.
    """
    # --- Process TARC Data ---
    # Create a DataFrame for TARC
    df_tarc = pd.DataFrame(tarc_actions, columns=['steering', 'throttle', 'action_duration'])
    df_tarc['cumulative_steps'] = tarc_steps

    # Calculate individual durations for each action
    df_tarc['action_duration'] = df_tarc['action_duration'].apply(compute_time)
    df_tarc = df_tarc.iloc[:-1]
    # Repeat each action row by its duration to upsample the data
    df_tarc_upsampled = df_tarc.loc[df_tarc.index.repeat(df_tarc['action_duration'])]
    df_tarc_upsampled['step'] = range(len(df_tarc_upsampled))

    # --- Process Baseline Data ---
    df_baseline = pd.DataFrame(baseline_actions, columns=['steering', 'throttle'])
    df_baseline['step'] = baseline_steps
    df_baseline = df_baseline.iloc[:-1]
    # --- Calculate Cumulative Effort for Both ---
    # 1. Calculate the squared action magnitude at each step
    df_tarc_upsampled['sq_action'] = np.sum(df_tarc_upsampled[['steering', 'throttle']] ** 2, axis=1)
    df_baseline['sq_action'] = np.sum(df_baseline[['steering', 'throttle']] ** 2, axis=1)

    # 2. Calculate the cumulative sum of this cost over the episode
    df_tarc_upsampled['cumulative_cost'] = df_tarc_upsampled['sq_action'].cumsum()
    df_baseline['cumulative_cost'] = df_baseline['sq_action'].cumsum()

    tarc_action_deltas = df_tarc_upsampled[['steering', 'throttle']].diff()
    df_tarc_upsampled['action_delta'] = np.linalg.norm(tarc_action_deltas.values, axis=1)

    baseline_action_deltas = df_baseline[['steering', 'throttle']].diff()
    df_baseline['action_delta'] = np.linalg.norm(baseline_action_deltas.values, axis=1)
    # --- Plotting ---
    plt.style.use(['science', 'ieee'])
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

    # Plot 1: Cumulative Control Effort
    ax1.plot(df_tarc_upsampled['step'], df_tarc_upsampled['cumulative_cost'], label='TARC Policy', color='royalblue',
             linewidth=2.5)
    ax1.plot(df_baseline['step'], df_baseline['cumulative_cost'], label='Baseline Policy', color='red', linewidth=2.5,
             alpha=0.8)
    ax1.set_title('A) Cumulative Control Effort', fontsize=14, loc='left')
    ax1.set_xlabel('Control Step', fontsize=12)
    ax1.set_ylabel('Cumulative Sum of Squared Actions', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True)
    ax1.set_xlim(0, max(len(df_tarc_upsampled), len(df_baseline)))
    ax1.set_ylim(bottom=0)

    # Plot 2: Action Smoothness (Deltas)
    delta_data = [df_baseline['action_delta'].dropna(), df_tarc_upsampled['action_delta'].dropna()]
    labels = ['Baseline', 'TARC']
    bplot = ax2.boxplot(delta_data, vert=True, patch_artist=True, labels=labels,
                        showfliers=False)  # showfliers=False makes the plot cleaner

    colors = ['lightcoral', 'lightblue']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

    ax2.set_title('B) Action Smoothness (Deltas)', fontsize=14, loc='left')
    ax2.set_ylabel('Magnitude of Action Change', fontsize=12)
    ax2.grid(True, axis='y')

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('RCCar_action_performance.png', dpi=300)
    print(f"\nPlot saved successfully to 'RCCar_action_performance.png'")
    plt.show()

def plot_actions(actions, steps, title="Actions", scale_factor=2.0):
    # Create subplots with increased size and spacing
    fig, axes = plt.subplots(nrows=1, ncols=2,
                             figsize=(scale_factor * 15, scale_factor * 8))  # Larger figure size
    axes = axes.flatten()

    # Common styling parameters
    plot_config = {
        "linewidth": 2.5,
        "fontsize_labels": 16,
        "fontsize_ticks": 14,
        "spine_width": 1.2,
        "wspace": 0.4  # Horizontal space between subplots
    }

    # Plot steering angle
    axes[0].plot(steps, actions[:, 0], label="Steering", color="b", linewidth=plot_config["linewidth"])
    axes[0].set_xlabel("Step", fontsize=plot_config["fontsize_labels"])
    axes[0].set_ylabel("Steering Angle", fontsize=plot_config["fontsize_labels"])
    axes[0].set_title("Steering Angle", fontsize=18, pad=15)

    # Plot throttle
    axes[1].plot(steps, actions[:, 1], label="Throttle", color="g", linewidth=plot_config["linewidth"])
    axes[1].set_xlabel("Step", fontsize=plot_config["fontsize_labels"])
    axes[1].set_ylabel("Throttle", fontsize=plot_config["fontsize_labels"])
    axes[1].set_title("Throttle", fontsize=18, pad=15)

    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=plot_config["fontsize_ticks"])
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
        ax.legend(fontsize=plot_config["fontsize_labels"])

        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        for spine in ['bottom', 'left']:
            ax.spines[spine].set_linewidth(plot_config["spine_width"])

    fig.suptitle(title, fontsize=20, y=0.95)
    plt.subplots_adjust(wspace=plot_config["wspace"])
    plt.show()

def plot_combined_trajectory(states, rewards, actions, steps, run_reward=None, run_actions=None, title="Combined Training Trajectory", scale_factor=1.5, spline_smoothing=0.2):
    """
    Combines plots for all states in one subplot, all actions in one subplot, and a separate reward plot,
    with the current run's reward and action counts displayed within their respective plots.
    Uses spline interpolation for smoother lines.

    Args:
        states (np.ndarray): Array of state trajectories (num_steps, num_states).
        rewards (np.ndarray): Array of rewards (num_steps,).
        actions (np.ndarray): Array of actions (num_steps, num_actions).
        steps (np.ndarray): Array of step numbers (num_steps,).
        run_reward (float, optional): The total reward achieved in this specific run. Defaults to None.
        run_actions (int, optional): The total number of actions in this specific run. Defaults to None.
        title (str): Overall title of the figure.
        scale_factor (float): Scaling factor for the figure size.
        spline_smoothing (float): Smoothing factor for the spline interpolation (0 for no smoothing).
    """
    num_states = states.shape[1]
    num_actions = actions.shape[1]

    fig = plt.figure(figsize=(scale_factor * 15, scale_factor * 7))  # Slightly reduced figure height
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[2, 1])  # Adjust grid for layout

    plot_config = {
        "linewidth": 1.0,
        "fontsize_labels": 10,
        "fontsize_ticks": 8,
        "spine_width": 0.6,
        "wspace": 0.3,
        "hspace": 0.4,
        "annotation_fontsize": 9
    }

    # Plot all States in one subplot (gs[0, 0])
    state_labels = ['x', 'y', 'theta', 'velocity  x', 'velocity y', 'angular velocity']
    ax_states = fig.add_subplot(gs[0, 0])
    for i in range(num_states):
        spl = UnivariateSpline(steps, states[:, i], s=spline_smoothing)
        states_smooth = spl(steps)
        ax_states.plot(steps, states_smooth, label=state_labels[i], linewidth=plot_config["linewidth"])
    ax_states.set_xlabel("Step", fontsize=plot_config["fontsize_labels"])
    ax_states.set_ylabel("State Values", fontsize=plot_config["fontsize_labels"])
    ax_states.tick_params(axis='both', which='major', labelsize=plot_config["fontsize_ticks"])
    ax_states.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
    ax_states.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax_states.legend(fontsize=plot_config["fontsize_labels"] - 2)
    for s in steps:
        ax_states.axvline(x=s, color='gray', linestyle='--', linewidth=0.5, zorder=2)
    for spine in ['top', 'right']:
        ax_states.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax_states.spines[spine].set_linewidth(plot_config["spine_width"])
    ax_states.set_title("States", fontsize=12)

    # Plot Rewards in one subplot (gs[0, 1])
    ax_reward = fig.add_subplot(gs[0, 1])
    spl = UnivariateSpline(steps, rewards, s=spline_smoothing)
    rewards_smooth = spl(steps)
    ax_reward.plot(steps, rewards_smooth, label="Reward", color="g", linewidth=plot_config["linewidth"])
    ax_reward.set_xlabel("Step", fontsize=plot_config["fontsize_labels"])
    ax_reward.set_ylabel("Reward", fontsize=plot_config["fontsize_labels"])
    ax_reward.tick_params(axis='both', which='major', labelsize=plot_config["fontsize_ticks"])
    ax_reward.xaxis.set_major_locator(mticker.MaxNLocator(nbins=3, integer=True))
    ax_reward.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
    ax_reward.legend(fontsize=plot_config["fontsize_labels"])
    for s in steps:
        ax_reward.axvline(x=s, color='gray', linestyle='--', linewidth=0.5, zorder=2)
    for spine in ['top', 'right']:
        ax_reward.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax_reward.spines[spine].set_linewidth(plot_config["spine_width"])
    ax_reward.set_title("Running Reward", fontsize=12)

    # Add reward information (current run only)
    reward_info_text = ""
    if run_reward is not None:
        reward_info_text += f"Run Reward: {run_reward:.2f}"

    if reward_info_text:
        ax_reward.annotate(reward_info_text,
                           xy=(0.05, 0.85), xycoords='axes fraction',
                           fontsize=plot_config["annotation_fontsize"],
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='lightgray', boxstyle='round,pad=0.3'))

    # Plot all Actions in one subplot (gs[1, :])
    action_labels = ['steering', 'throttle']
    ax_actions = fig.add_subplot(gs[1, :])
    action_steps = steps[:-1] if len(steps) == len(actions) + 1 else steps
    for i in range(num_actions):
        spl = UnivariateSpline(action_steps, actions[:, i], s=spline_smoothing)
        actions_smooth = spl(action_steps)
        ax_actions.plot(action_steps, actions_smooth, label=action_labels[i], color=f'C{i+1}', linewidth=plot_config["linewidth"]) # Use different colors
    ax_actions.set_xlabel("Step", fontsize=plot_config["fontsize_labels"])
    ax_actions.set_ylabel("Action Values", fontsize=plot_config["fontsize_labels"])
    ax_actions.tick_params(axis='both', which='major', labelsize=plot_config["fontsize_ticks"])
    ax_actions.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5, integer=True))
    ax_actions.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax_actions.legend(fontsize=plot_config["fontsize_labels"] - 2)
    for s in action_steps:
        ax_actions.axvline(x=s, color='gray', linestyle='--', linewidth=0.5, zorder=2)
    for spine in ['top', 'right']:
        ax_actions.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax_actions.spines[spine].set_linewidth(plot_config["spine_width"])
    ax_actions.set_title("Actions", fontsize=12)

    # Add action information (current run only)
    action_info_text = ""
    if run_actions is not None:
        action_info_text += f"Run Actions: {run_actions}"

    if action_info_text:
        ax_actions.annotate(action_info_text,
                            xy=(0.05, 0.85), xycoords='axes fraction',
                            fontsize=plot_config["annotation_fontsize"],
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='lightgray', boxstyle='round,pad=0.3'))

    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=0.5, w_pad=0.5)
    plt.show()


# 3. DEFINE WHICH STATES TO PLOT
#    Map the desired plot label to the actual column name/index in your CSV.
STATES_TO_PLOT = {
    'x': 'Obs_0',
    'y': 'Obs_1',
    'theta': 'Obs_2',
    'velocity x': 'Obs_3',
    'velocity y': 'Obs_4',
    'angular velocity': 'Obs_5'
}

# 4. PLOTTING AND SMOOTHING CONFIGURATION
SPLINE_SMOOTHING_FACTOR = 0.1

# This dictionary holds all the styling parameters for the plot.
PLOT_CONFIG = {
    "scale_factor": 1.5,
    "linewidth": 1.5,
    "fontsize_labels": 12,
    "fontsize_ticks": 10,
    "fontsize_title": 14,
    "fontsize_legend": 10,
    "annotation_fontsize": 10,
    "spine_width": 1.0
}


# --- End Configuration ---

def plot_states_and_rewards(time_data, state_array, reward_data, state_labels, spline_smoothing, plot_config):
    """
    Creates a combined plot with state trajectories on the left and the reward
    trajectory on the right, taking NumPy arrays as input.
    """
    print("Generating combined state and reward plot...")

    # Create a figure with a specific grid layout
    fig, ax_states = plt.subplots(figsize=(plot_config["scale_factor"] * 8, plot_config["scale_factor"] * 5))

    # --- 1. Plot all States in the left subplot ---

    for i, label in enumerate(state_labels):
        state_trajectory = state_array[:, i]
        try:
            spline = UnivariateSpline(time_data, state_trajectory, s=spline_smoothing)
            time_smooth = np.linspace(time_data.min(), time_data.max(), 500)
            state_smooth = spline(time_smooth)
            ax_states.plot(time_smooth, state_smooth, label=label, linewidth=plot_config["linewidth"])
        except Exception as e:
            print(f"Could not create spline for '{label}'. Plotting raw data. Error: {e}")
            ax_states.plot(time_data, state_trajectory, label=label, linewidth=plot_config["linewidth"])

    # Add vertical dashed lines for each control step
    for t in time_data:
        ax_states.axvline(x=t, color='gray', linestyle='--', linewidth=0.5, alpha=0.6, zorder=-1)

    # Formatting for the states plot
    ax_states.set_title("Tacos5: State trajectory", fontsize=plot_config["fontsize_title"])
    ax_states.set_xlabel("Time (s)", fontsize=plot_config["fontsize_labels"])
    ax_states.set_ylabel("State Values", fontsize=plot_config["fontsize_labels"])
    ax_states.tick_params(axis='both', which='major', labelsize=plot_config["fontsize_ticks"])
    ax_states.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7, integer=False))
    ax_states.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax_states.legend(fontsize=plot_config["fontsize_legend"])
    ax_states.spines['top'].set_visible(False)
    ax_states.spines['right'].set_visible(False)
    ax_states.set_xlim(time_data.min(), time_data.max())

    fig.tight_layout()  # Adjust rect to make space for suptitle

    # Save the figure in high-quality PDF format
    output_filename = 'plot_states.png'
    fig.savefig(output_filename, dpi=300)
    print(f"Plot saved as: {output_filename}")

    plt.show()

#we get the evaluation data from wandb for the runs
def fetch_run_data(project_name, tag, penalize):
    """
    Fetches logged rewards and number of actions from Wandb runs with a specific tag.

    Args:
        project_name (str): The name of your Wandb project.
        tag (str): The tag used to identify the runs of interest.
        penalize: Bool which defines whether we penalize the interactions on the reward

    Returns:
        tuple: A tuple containing two lists:
               - all_rewards (list of float): A list of total rewards from each matching run.
               - all_num_actions (list of int): A list of total actions from each matching run.
               Returns empty lists if no matching runs are found or if the data is not logged.
    """
    api = wandb.Api()
    runs = api.runs(project_name, {"tags":tag})

    all_rewards = []
    all_num_actions = []

    for run in runs:
        summary = run.summary
        if "total reward" in summary and "number of actions" in summary:
            if 'terminal reward' in summary.keys() and summary['terminal reward'] != 0.0: ## we use the "normal" unadaptive control frequency, so that results make more sense
                summary['number of actions'] = base_control_frequency[tag]
            elif 'terminal_reward' in summary.keys() and summary['terminal_reward'] != 0.0:
                summary['number of actions'] = base_control_frequency[tag]
            all_rewards.append(summary["total reward"] - int(penalize) * 0.1 * summary['number of actions']) #we penalize using the interaction cost 0.1 from training
            all_num_actions.append(summary["number of actions"])
        elif "total_reward" in summary and "number of actions" in summary:
            if 'terminal reward' in summary.keys() and summary['terminal reward'] != 0.0: ## we use the "normal" unadaptive control frequency, so that results make more sense
                summary['number of actions'] = base_control_frequency[tag]
            elif 'terminal_reward' in summary.keys() and summary['terminal_reward'] != 0.0:
                summary['number of actions'] = base_control_frequency[tag]
            all_rewards.append(summary["total_reward"])
            all_num_actions.append(summary["number of actions"])
        else:
            print(f"Warning: Run {run.name} with tag '{tag}' does not contain 'total_reward' or 'number of actions' in its summary.")

    all_avg_frequency = [x*30.0 / 200.0 for x in all_num_actions] ## we convert to average control frequency
    return all_rewards, all_avg_frequency

def analyze_run_data(all_rewards, all_num_actions):
    """
    Computes the average and standard deviation of rewards and the average and standard deviation of the number of actions.

    Args:
        all_rewards (list of float): A list of total rewards.
        all_num_actions (list of int): A list of total number of actions.

    Returns:
        tuple: A tuple containing:
               - avg_reward (float or None): Average total reward, None if no rewards.
               - std_reward (float or None): Standard deviation of total reward, None if fewer than 2 rewards.
               - avg_num_actions (float or None): Average total number of actions, None if no action counts.
               - std_num_actions (float or None): Standard deviation of total number of actions, None if fewer than 2 action counts.
    """
    avg_reward = np.mean(all_rewards) if all_rewards else None
    stderr_reward = sem(all_rewards) if len(all_rewards) >= 2 else None
    avg_num_actions = np.mean(all_num_actions) if all_num_actions else None
    stderr_num_actions = sem(all_num_actions) if len(all_num_actions) >= 2 else None
    return avg_reward, stderr_reward, avg_num_actions, stderr_num_actions

def get_wandb_run_stats(project_name, tag, penalize):
    """
    Fetches runs from Wandb based on a tag and computes the average and standard
    deviation of logged total rewards and the average and standard deviation of total actions.

    Args:
        project_name (str): The name of your Wandb project.
        tag (str): The tag used to identify the runs of interest.

    Returns:
        tuple: A tuple containing:
               - avg_reward (float or None): Average total reward.
               - std_reward (float or None): Standard deviation of total reward.
               - avg_num_actions (float or None): Average total number of actions.
               - std_num_actions (float or None): Standard deviation of total number of actions.
    """
    all_rewards, all_num_actions = fetch_run_data(project_name, tag, penalize)
    avg_reward, std_reward, avg_num_actions, std_num_actions = analyze_run_data(all_rewards, all_num_actions)

    print(f"Analysis for runs with tag '{tag}':")
    if avg_reward is not None:
        print(f"  Average Total Reward: {avg_reward:.2f}")
    else:
        print("  No reward data found for the specified tag.")

    if std_reward is not None:
        print(f"  Standard Deviation of Total Reward: {std_reward:.2f}")
    else:
        print("  Standard deviation of reward cannot be computed (less than 2 data points).")

    if avg_num_actions is not None:
        print(f"  Average Total Number of Actions: {avg_num_actions:.2f}")
    else:
        print("  No action count data found for the specified tag.")

    if std_num_actions is not None:
        print(f"  Standard Deviation of Total Number of Actions: {std_num_actions:.2f}")
    else:
        print("  Standard deviation of action count cannot be computed (less than 2 data points).")


    return avg_reward, std_reward, avg_num_actions, std_num_actions

angle_idx = 2
PENALIZE = False
plt.style.use(['science', 'ieee'])

mean_rewards = []
std_rewards = []
mean_actions = []
std_actions = []
## PPO
file_path = "../../results/rccar/trajectories/v2/origin/PPO/trajectory_hj6j34jo.npy"
trajectory = np.load(file_path, allow_pickle=True)
trajectory = correct_all_duplicate_pairs(trajectory)
state = trajectory[:, 0]
state = np.stack(state)
action = np.array(trajectory[:, 1])
action[200] = action[200][:2] #clip the last action
action = np.stack(action)
reward = trajectory[:, 2]
step = trajectory[:, 3]
ppo_steps = step
ppo_actions = action

# retrieve angle theta from sin(theta), cos(theta)
decoded_states = [decode_angles_numpy(s, angle_idx=2) for s in state]
decoded_states = np.array(decoded_states)
rotated_state = rotate_coordinates(decoded_states, encode_angle=False)

# retrieve run data from wandb
#avg_reward, std_reward, avg_actions, std_action = get_wandb_run_stats('arnavsukhija-eth-zurich/TaCosFinalResultsv2_origin', 'ppo', PENALIZE)
#plot_combined_trajectory(rotated_state,reward, action[:, :2], step, title="PPO RC car trajectory", run_actions=200, run_reward=93.41553)

#mean_rewards.append(avg_reward)
#std_rewards.append(std_reward)
#mean_actions.append(avg_actions)
#std_actions.append(std_action)

## PPO tacos 2
file_path = "../../results/rccar/trajectories/v2/origin/TARC2/trajectory_3ft20fzi.npy"
trajectory = np.load(file_path, allow_pickle=True)
trajectory = correct_all_duplicate_pairs(trajectory)
state = np.stack(trajectory[:, 0])
action = np.stack(trajectory[:, 1])
reward = trajectory[:, 2]
steps = trajectory[:, 3]

decoded_states = [decode_angles_numpy(s, angle_idx=angle_idx) for s in state]
decoded_states = np.array(decoded_states)
rotated_states = rotate_coordinates(decoded_states, encode_angle=False)

#avg_reward, std_reward, avg_actions, std_action = get_wandb_run_stats('arnavsukhija-eth-zurich/TaCosFinalResultsv2_origin', 'tacos2_hardware', PENALIZE)
#plot_combined_trajectory(rotated_states ,reward, action[:, :2], steps, title="PPO-Tacos2 RC car trajectory", run_actions=200, run_reward=96.0051)


### PPO tacos 3
file_path = "../../results/rccar/trajectories/v2/origin/TARC3/trajectory_wzl6uv83.npy"
trajectory = np.load(file_path, allow_pickle=True)
trajectory = correct_all_duplicate_pairs(trajectory)
state = np.stack(trajectory[:, 0])
action = np.stack(trajectory[:, 1])
reward = trajectory[:, 2]
steps = trajectory[:, 3]
decoded_states = [decode_angles_numpy(s, angle_idx=angle_idx) for s in state]
decoded_states = np.array(decoded_states)
rotated_states = rotate_coordinates(decoded_states, encode_angle=False)
#avg_reward, std_reward, avg_actions, std_action = get_wandb_run_stats('arnavsukhija-eth-zurich/TaCosFinalResultsv2_origin', 'hardware_3actions', PENALIZE)
#plot_combined_trajectory(rotated_states ,reward, action[:, :2], steps, title="PPO-Tacos3 RC car trajectory", run_actions=100, run_reward=127.13218, spline_smoothing=0.3)

#mean_rewards.append(avg_reward)
#std_rewards.append(std_reward)
#mean_actions.append(avg_actions)
#std_actions.append(std_action)

## Tacos 4
file_path = "../../results/rccar/trajectories/v2/origin/TARC4/trajectory_g9hv0sfq.npy"
trajectory = np.load(file_path, allow_pickle=True)
trajectory = correct_all_duplicate_pairs(trajectory)
state = trajectory[:, 0]
state = np.stack(state)
action = np.array(trajectory[:, 1])
action = np.stack(action)
reward = trajectory[:, 2]
step = trajectory[:, 3]
plot_action_deltas(action, step, ppo_actions, ppo_steps, 'RC Car Action Delta')
exit()

# retrieve angle theta from sin(theta), cos(theta)
decoded_states = [decode_angles_numpy(s, angle_idx=2) for s in state]
decoded_states = np.array(decoded_states)
rotated_state = rotate_coordinates(decoded_states, encode_angle=False)

# retrieve run data from wandb
#avg_reward, std_reward, avg_actions, std_action = get_wandb_run_stats('arnavsukhija-eth-zurich/TaCosFinalResultsv2_origin', 'hardware_4actions', PENALIZE)
#plot_combined_trajectory(rotated_state,reward, action[:, :2], step, title="PPO-Tacos4 RC car trajectory", run_actions=67, run_reward=115.31706, spline_smoothing=0.4)

#mean_rewards.append(avg_reward)
#std_rewards.append(std_reward)
#mean_actions.append(avg_actions)
#std_actions.append(std_action)

##Tacos 5
file_path = "final Trajectorie v2/goal/"
trajectory = np.load(file_path, allow_pickle=True)
trajectory = correct_all_duplicate_pairs(trajectory)
state = trajectory[:, 0]
state = np.stack(state)
action = np.array(trajectory[:, 1])
action = np.stack(action)
reward = trajectory[:, 2]
step = trajectory[:, 3]

# retrieve angle theta from sin(theta), cos(theta)
decoded_states = [decode_angles_numpy(s, angle_idx=2) for s in state]
decoded_states = np.array(decoded_states)
rotated_state = rotate_coordinates(decoded_states, encode_angle=False)

# retrieve run data from wandb
avg_reward, std_reward, avg_actions, std_action = get_wandb_run_stats('arnavsukhija-eth-zurich/TaCosFinalResultsv2_origin', 'tacos5_hardware', PENALIZE)
#lot_combined_trajectory(rotated_state,reward, action[:, :2], step, title="PPO-Tacos5 RC car trajectory", run_actions=50, run_reward=110.88953, spline_smoothing=0.5)
time_data = step * 1.0 / 30
state_labels = list(STATES_TO_PLOT.keys())
#plot_states_and_rewards(time_data, rotated_state, reward, state_labels, SPLINE_SMOOTHING_FACTOR, PLOT_CONFIG)

mean_rewards.append(avg_reward)
std_rewards.append(std_reward)
mean_actions.append(avg_actions)
std_actions.append(std_action)

# Tacos 10
file_path = "../../results/rccar/trajectories/v2/origin/TARC10/trajectory_rirs1len.npy"
trajectory = np.load(file_path, allow_pickle=True)
trajectory = correct_all_duplicate_pairs(trajectory)
state = trajectory[:, 0]
state = np.stack(state)
action = np.array(trajectory[:, 1])
action = np.stack(action)
reward = np.array(trajectory[:, 2])
step = np.array(trajectory[:, 3])
# retrieve angle theta from sin(theta), cos(theta)
decoded_states = [decode_angles_numpy(s, angle_idx=2) for s in state]
decoded_states = np.array(decoded_states)
rotated_state = rotate_coordinates(decoded_states, encode_angle=False)

# retrieve run data from wandb
avg_reward, std_reward, avg_actions, std_action = get_wandb_run_stats('arnavsukhija-eth-zurich/TaCosFinalResultsv2_origin', 'tacos10_hardware', PENALIZE)
#plot_combined_trajectory(rotated_state,reward, action[:, :2], step, title="PPO-Tacos10 RC car trajectory", run_actions=26, run_reward=77.1638, spline_smoothing=0.7)

mean_rewards.append(avg_reward)
std_rewards.append(std_reward)
mean_actions.append(avg_actions)
std_actions.append(std_action)


algorithm_names = ['Baseline', 'Tacos3', 'Tacos4', 'Tacos5', 'Tacos10']

data_to_plot = [
    {"data": mean_rewards, "title": "Distribution of Mean Rewards", "xlabel": "Mean Reward"},
    {"data": std_rewards, "title": "Deviation of Rewards", "xlabel": "Reward std"},
    {"data": mean_actions, "title": "Distribution of Mean Number of Interactions", "xlabel": "Mean Number of Interactions"},
    {"data": std_actions, "title": "Deviation of Number of Interactions", "xlabel": "Interaction std"}
]

x_positions = np.arange(len(algorithm_names))  # Create numerical positions for the bars
bar_width = 0.7  # Adjust the width of the bars as needed


def create_reward_plot(x_pos, names, means, stds, penalize):
        colors = sns.color_palette("colorblind")
        fig, ax = plt.subplots()

        ax.bar(x_pos, means, yerr=stds, capsize=3, alpha=0.75, color=colors[0])

        ax.set_ylabel("Total Reward")
        if penalize:
            ax.set_title("Total Reward (penalized)")
        else:
            ax.set_title("Total Reward (unpenalized)")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha="right")

        # Save the figure as a high-quality PDF
        output_filename = 'bar_plot_total_reward_Rccar_penalized.pdf' if penalize else 'bar_plot_total_reward_Rccar.pdf'
        fig.tight_layout()
        fig.savefig(output_filename)
        print(f"Plot saved as: {output_filename}")
        plt.close(fig)  # Close the figure to free up memory


def create_frequency_plot(x_pos, names, means, stds, min_freqs):
        colors = sns.color_palette("colorblind")
        fig, ax = plt.subplots()
        bar_width = 0.7

        ax.bar(x_pos, means, width=bar_width, yerr=stds, capsize=3, alpha=0.75,label='Average Frequency', color=colors[1])

        # Add the horizontal dashed lines for the theoretical minimum
        for i, pos in enumerate(x_pos):
            h_label = 'Frequency Lower Bound' if i == 0 else None
            ax.hlines(y=min_freqs[i], xmin=pos - bar_width / 2, xmax=pos + bar_width / 2,
                      colors='black', linestyles='dashed', lw=1.5, label=h_label)

        ax.set_ylabel("Average Control Frequency (Hz)")
        ax.set_title("Average Control Frequency")
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha="right")
        ax.legend()

        # Save the figure as a high-quality PDF
        output_filename = 'bar_plot_average_frequency_Rccar.pdf'
        fig.tight_layout()
        fig.savefig(output_filename)
        print(f"Plot saved as: {output_filename}")
        plt.close(fig)

create_reward_plot(x_positions, algorithm_names, mean_rewards, std_rewards, PENALIZE)
create_frequency_plot(x_positions, algorithm_names, mean_actions,std_actions, min_freqs=[30, 10, 7.5, 6, 3])

# Tacos 5 higher throttle
file_path = "../../results/rccar/trajectories/v1/goal_origin_max_throttle_0.4/trajectory_cgogh2o2.npy"
trajectory = np.load(file_path, allow_pickle=True)
trajectory = correct_all_duplicate_pairs(trajectory)
state = trajectory[:, 0]
state = np.stack(state)
action = np.array(trajectory[:, 1])
action = np.stack(action)
reward = trajectory[:, 2]
step = trajectory[:, 3]

# retrieve angle theta from sin(theta), cos(theta)
decoded_states = [decode_angles_numpy(s, angle_idx=2) for s in state]
decoded_states = np.array(decoded_states)
rotated_state = rotate_coordinates(decoded_states, encode_angle=False)
print(step)
# retrieve run data from wandb
avg_reward, std_reward, avg_actions, std_action = get_wandb_run_stats('arnavsukhija-eth-zurich/TaCosFinalResultsv2_origin', 'tacos5_hardware', PENALIZE)
#plot_combined_trajectory(rotated_state,reward, action[:, :2], step, title="PPO-Tacos5 RC car trajectory with higher throttle", run_actions=50, run_reward=82.16157, spline_smoothing=0.5)


##we use before DR and after D

# --- Configuration ---
TARC_FILE_PATH = "../../results/rccar/trajectories/v2/origin/TARC5/trajectory_xx35nmsz.npy"
BASELINE_FILE_PATH = "../../results/rccar/trajectories/v2/origin/PPO/trajectory_72pmrwwo.npy"
OUTPUT_PLOT_FILENAME = 'rc_car_action_cost.png'


def process_trajectory_file(file_path, is_tarc=False):
    """
    Loads a trajectory .npy file and processes it into a pandas DataFrame.
    For TARC, it correctly upsamples the sparse actions.
    """
    # Load the NumPy array
    trajectory = np.load(file_path, allow_pickle=True)
    trajectory = correct_all_duplicate_pairs([trajectory])
    # Extract action and step data
    actions = np.stack(trajectory[:, 1])
    steps = np.stack(trajectory[:, 3])

    df = pd.DataFrame(actions, columns=['steering', 'throttle'])
    df['cumulative_steps'] = steps

    # --- Upsample TARC data for a fair comparison ---
    if is_tarc:
        # Calculate individual durations for each action
        df['duration'] = df['cumulative_steps'].diff().fillna(1).astype(int)

        # Repeat each action row by its duration
        df = df.loc[df.index.repeat(df.duration)]

        # Create a new step index for the upsampled data
        df['step'] = range(len(df))
        df = df.set_index('step')
    else:
        # For the baseline, the step is just the row index
        df['step'] = df.index

    # Calculate the squared action magnitude at each step
    df['sq_action'] = np.sum(df[['steering', 'throttle']] ** 2, axis=1)

    # Calculate the cumulative sum of this cost over the episode
    df['cumulative_sq_action'] = df['sq_action'].cumsum()

    return df
