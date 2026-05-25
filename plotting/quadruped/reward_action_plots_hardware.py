import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import scienceplots


def process_all_data(agents, tasks, switch_costs):
    """
    Processes all experimental data from CSVs and returns a structured dictionary.
    """
    print("--- Processing all experimental data... ---")
    all_data = {}
    for agent_name in agents:
        agent_data = {}
        path_fn = None
        if agent_name == 'PPO':
            def get_ppo_path(seed):
                return f'../sim2realResults/PPO/ppo_seed{seed}.csv'

            path_fn = get_ppo_path
        else:
            def get_tacos_path(agent, seed):
                path = f'../sim2realResults/Tacos/{agent}/seed{seed}/{agent}_seed{seed}.csv'
                return path if os.path.exists(path) else None

            path_fn = lambda seed: get_tacos_path(agent_name, seed)

        for task_name_key, task_name_csv in tasks.items():
            task_runs = []
            for seed in range(5):
                path = path_fn(seed)
                if path is None:
                    print(f"Warning: Path not found for {agent_name} seed {seed}. Skipping.")
                    continue
                try:
                    df = pd.read_csv(path)
                    df_task = df[df['ModeDetails'] == task_name_csv].iloc[:, 2:].mean(axis=0)
                    task_runs.append(df_task)
                except (FileNotFoundError, pd.errors.EmptyDataError):
                    continue

            if not task_runs:
                print(f"Warning: No data for {agent_name}, task {task_name_key}. Check paths and CSV content.")
                continue

            df_seeds = pd.DataFrame(task_runs)
            df_seeds['Average Control Frequency'] = df_seeds['Total Action Switches'] / (df_seeds['Env Steps'] * 0.02)
            switch_cost = switch_costs.get(agent_name, 0)
            df_seeds['Total Reward (penalized)'] = df_seeds['Total Reward'] - switch_cost * df_seeds[
                'Total Action Switches']
            df_seeds['Total Reward (unpenalized)'] = df_seeds['Total Reward']
            summary = pd.DataFrame({'Mean': df_seeds.mean(axis=0), 'SEM': df_seeds.sem(axis=0)})
            agent_data[task_name_key] = summary
        all_data[agent_name] = agent_data
    print("--- Data processing complete. ---")
    return all_data


def create_final_results_plot(data, tasks, min_freqs):
    """
    Generates the final, publication-quality 1x3 plot figure with a professional aspect ratio.
    """
    print("--- Generating final results plot... ---")

    plt.style.use(['science', 'ieee'])

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5))

    colors = sns.color_palette('colorblind', n_colors=len(tasks))
    agents = ['PPO', 'Tacos3', 'Tacos4', 'Tacos5', 'Tacos10']
    plot_agents = ['Baseline', 'TARC-3', 'TARC-4', 'TARC-5', 'TARC-10']
    task_keys = list(tasks.keys())

    plot_specs = [
        {'metric': 'Total Reward (penalized)', 'title': '(a) Penalized Reward'},
        {'metric': 'Total Reward (unpenalized)', 'title': '(b) Unpenalized Reward'},
        {'metric': 'Average Control Frequency', 'title': '(c) Avg. Control Frequency'}
    ]

    for i, spec in enumerate(plot_specs):
        ax = axes[i]
        metric_name = spec['metric']

        means = {task: [data.get(agent, {}).get(task, {}).get('Mean', {}).get(metric_name, np.nan) for agent in agents]
                 for task in task_keys}
        sems = {task: [data.get(agent, {}).get(task, {}).get('SEM', {}).get(metric_name, 0) for agent in agents] for
                task in task_keys}

        x = np.arange(len(agents))
        width = 0.27
        multiplier = -1

        for j, task_name in enumerate(task_keys):
            measurement = means[task_name]
            error = sems[task_name]
            offset = width * multiplier
            ax.bar(x + offset, measurement, width, label=task_name, yerr=error,
                   capsize=2, color=colors[j], linewidth=0.6,
                   error_kw={'elinewidth': 0.8})
            multiplier += 1
            if metric_name == 'Average Control Frequency':
                for k, agent_name in enumerate(agents):
                    if agent_name == 'PPO':
                        continue
                    ax.hlines(y=min_freqs[k],
                              xmin=x[k] - 3*width / 2,
                              xmax=x[k] + 3*width / 2,
                              colors='black', linestyles='dashed', lw=1.5)

        ax.set_title(spec['title'])
        ax.set_xticks(x)
        ax.set_xticklabels(plot_agents, rotation=45, ha="right")


    axes[0].set_ylabel("Total Reward")
    axes[2].set_ylabel("Frequency (Hz)")
    axes[1].set_ylabel(None)

    from matplotlib.lines import Line2D
    handles, labels = axes[0].get_legend_handles_labels()
    dashed_line = Line2D([0], [0], color='black', lw=1.5, linestyle='--', label='Frequency Lower Bound')
    handles.append(dashed_line)
    labels.append('Frequency Lower Bound')
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=len(tasks) + 1, frameon=False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_filename = 'Go1_Results_plots.pdf'
    plt.savefig(output_filename)
    print(f"--- Final plot saved as: {output_filename} ---")


def create_video_plot(data):
    """
    --- MODIFIED FUNCTION ---
    Generates a 1x2 plot for the video, comparing Baseline and TARC-4
    for the "Run Then Turn" scenario.
    """
    print("--- Generating video results plot... ---")

    plt.style.use(['science', 'ieee'])

    # --- CHANGE: Create a 1x2 subplot figure ---
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.5))

    colors = sns.color_palette('colorblind', n_colors=2)

    # --- CHANGE: Specify only the agents and task for the video ---
    agents = ['PPO', 'Tacos4']
    plot_agents = ['Baseline', 'TARC']
    task_key = 'Run Then Turn'

    plot_specs = [
        {'metric': 'Total Reward (unpenalized)', 'title': 'Total Reward'},
        {'metric': 'Average Control Frequency', 'title': 'Avg. Control Frequency'}
    ]

    for i, spec in enumerate(plot_specs):
        ax = axes[i]
        metric_name = spec['metric']

        # Extract mean and error for the two agents
        means = [data.get(agent, {}).get(task_key, {}).get('Mean', {}).get(metric_name, np.nan) for agent in agents]
        sems = [data.get(agent, {}).get(task_key, {}).get('SEM', {}).get(metric_name, 0) for agent in agents]

        ax.bar(plot_agents, means, yerr=sems, color=colors, capsize=3, width=0.6)

        ax.set_title(spec['title'])
        ax.tick_params(axis='x', rotation=0)  # No rotation needed for two bars



    axes[0].set_ylabel("Total Reward")
    axes[1].set_ylabel("Frequency (Hz)")

    fig.tight_layout()

    output_filename = 'tarc_go1Results.png'
    plt.savefig(output_filename)
    print(f"--- Video plot saved as: {output_filename} ---")


# --- Main Execution ---
if __name__ == '__main__':
    agents_to_process = ['PPO', 'Tacos4']
    tasks_to_process = {
        'Run Then Turn': 'Fixed Sequence: run_then_turn_20s'
    }
    switch_costs_for_agents = {agent: 0.005 for agent in agents_to_process}
    min_freqs = [50, 50/3.0, 50/4.0, 50/5.0, 50/10.0]

    processed_data = process_all_data(agents_to_process, tasks_to_process, switch_costs_for_agents)
    create_video_plot(processed_data)