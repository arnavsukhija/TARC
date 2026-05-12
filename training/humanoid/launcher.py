import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import training_g1
from training.euler_util import generate_run_commands, generate_base_command, dict_permutations, available_gpus

################################################
################## G1 Humanoid ################
################################################

g1_tarc_runs = {
    'env_name': ['G1JoystickFlatTerrain'],
    'backend': ['generalized'],
    'project_name': ['G1_TARC'],
    'seed': list(range(5)),
    'switch_cost': [0.005],
    'max_time_repeat': [4, 5, 10],
    'min_time_repeat': [1],
    'time_as_part_of_state': [1],
    'num_final_evals': [1],
    'switch_cost_wrapper': [1],
}

g1_ppo_baselines = {
    'env_name': ['G1JoystickFlatTerrain'],
    'backend': ['generalized'],
    'project_name': ['G1_TARC'],
    'seed': list(range(5)),
    'base_dt_divisor': [1, 4, 5, 10],  # 50Hz, 12.5Hz, 10Hz, 5Hz
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
        cmd = generate_base_command(training_g1, flags=flags)
        command_list.append(cmd)

    for flags in dict_permutations(g1_ppo_baselines):
        cmd = generate_base_command(training_g1, flags=flags)
        command_list.append(cmd)

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
