import os
import sys

# Ensure the project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Temporarily set JAX to use CPU during launcher initialization on the login node 
# to prevent CUDA_ERROR_NO_DEVICE crashes.

import training_go1
from training.euler_util import generate_run_commands, generate_base_command, dict_permutations, available_gpus

################################################
#################### Quadruped ####################
################################################
go1_switch_cost = {'env_name': ['Go1JoystickFlatTerrain', ],
                     'backend': ['generalized', ],
                     'project_name': ["Go1_Stability_Rebuttal"],
                     'seed': list(range(5)),
                     'switch_cost': [0.005],
                     'max_time_repeat': [4,5,10],
                     'min_time_repeat': [1,],
                     'time_as_part_of_state': [1, ],
                     'num_final_evals': [1, ],
                      'perturb' : [1, ],
                      'switch_cost_wrapper': [1],
                     }

go1_ppo_baselines = {
    'env_name': ['Go1JoystickFlatTerrain'],
    'backend': ['generalized'],
    'project_name': ["Go1_Stability_Rebuttal"],
    'seed': list(range(5)),
    'perturb': [1],
    'base_dt_divisor': [1, 4, 5, 10], # 50Hz, 25Hz, 10Hz, 5Hz
    'switch_cost_wrapper': [0],
}

go1_tarc_runs = {
    'env_name': ['Go1JoystickFlatTerrain'],
    'backend': ['generalized'],
    'project_name': ["Go1_Sensitivity_Rebuttal"],
    'seed': list(range(5)),
    'switch_cost': [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1],
    'min_time_repeat': [1,],
    'max_time_repeat': [3, 4, 5, 10],
    'time_as_part_of_state': [1],
    'num_final_evals': [1],
    'perturb': [1],
    'switch_cost_wrapper': [1],
}

def main():
    command_list = []
    
    # Generate PPO baseline commands

    # Generate TARC commands
    for flags in dict_permutations(go1_tarc_runs):
        cmd = generate_base_command(training_go1, flags=flags)
        command_list.append(cmd)

    # submit jobs
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
