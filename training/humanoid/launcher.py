import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from training.euler_util import generate_run_commands, dict_permutations, available_gpus

TRAINING_SCRIPT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'training_g1.py'))


def generate_base_command(flags: dict) -> str:
    cmd = f"{sys.executable} -u {TRAINING_SCRIPT}"
    for flag, val in flags.items():
        cmd += f" --{flag}={val}"
    return cmd

################################################
################## G1 Humanoid ################
################################################

# Hardware deployment configs: switch_cost in [0.001, 0.005, 0.01], max_time_repeat in [4, 5].
# 0.001 added to test whether lower cost avoids the low-frequency collapse failure mode.
# Avoids the collapse zone (switch_cost=0.05 with max_time_repeat<=5) seen in sensitivity sweep.
# max_time_repeat capped at 5 (100ms hold at 50Hz) to preserve balance reactivity.
g1_tarc_runs = {
    'env_name': ['G1JoystickFlatTerrain'],
    'backend': ['generalized'],
    'project_name': ['G1_TARC'],
    'seed': list(range(5)),
    'switch_cost': [0.001, 0.005, 0.01],
    'max_time_repeat': [4, 5],
    'min_time_repeat': [1],
    'time_as_part_of_state': [1],
    'num_final_evals': [1],
    'switch_cost_wrapper': [1],
}

# Baseline: 50Hz (full frequency) and 10Hz (matches effective rate of max_time_repeat=5).
g1_ppo_baselines = {
    'env_name': ['G1JoystickFlatTerrain'],
    'backend': ['generalized'],
    'project_name': ['G1_TARC'],
    'seed': list(range(5)),
    'base_dt_divisor': [1, 5],  # 50Hz, 10Hz
    'switch_cost_wrapper': [0],
}

# Sensitivity sweep: vary switch cost and max hold duration
g1_sensitivity = {
    'env_name': ['G1JoystickFlatTerrain'],
    'backend': ['generalized'],
    'project_name': ['G1_TARC_Sensitivity'],
    'seed': list(range(5)),
    'switch_cost': [0.0001, 0.001, 0.005, 0.01, 0.05],
    'max_time_repeat': [3, 4, 5, 10],
    'min_time_repeat': [1],
    'time_as_part_of_state': [1],
    'num_final_evals': [1],
    'switch_cost_wrapper': [1],
}


def main():
    command_list = []

    for flags in dict_permutations(g1_tarc_runs):
        command_list.append(generate_base_command(flags))

    for flags in dict_permutations(g1_ppo_baselines):
        command_list.append(generate_base_command(flags))

    generate_run_commands(command_list,
                          num_cpus=1,
                          num_gpus=1,
                          gpu=available_gpus[3],
                          mode='euler',
                          duration='23:59:00',
                          prompt=True,
                          mem=32000)


if __name__ == '__main__':
    main()
