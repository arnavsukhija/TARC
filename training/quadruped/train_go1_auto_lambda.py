import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import training_go1
from training.euler_util import generate_run_commands, generate_base_command, dict_permutations, available_gpus

go1_auto_lambda = {
    'env_name': ['Go1JoystickFlatTerrain'],
    'backend': ['generalized'],
    'project_name': ['Go1_AutoLambda'],
    'seed': list(range(5)),
    'switch_cost_wrapper': [1],
    'use_auto_lambda': [1],
    'K_ratio': [0.3, 0.4, 0.6, 0.7],
    'alpha_lambda': [0.05],
    'lambda_init': [0.005],
    'lambda_min': [0.0],
    'lambda_max': [1.0],
    'min_time_repeat': [1],
    'max_time_repeat': [5, 10],
    'time_as_part_of_state': [1],
    'num_final_evals': [1],
    'perturb': [1],
}


def main():
    command_list = []

    for flags in dict_permutations(go1_auto_lambda):
        cmd = generate_base_command(training_go1, flags=flags)
        command_list.append(cmd)

    generate_run_commands(command_list,
                          num_cpus=1,
                          num_gpus=1,
                          gpu=available_gpus[3],
                          mode='euler',
                          duration='03:59:00',
                          prompt=True,
                          mem=32000)


if __name__ == '__main__':
    main()
