"""Launcher: retrain TARC-4 and PPO baselines with perturbations enabled.

Trains everything needed to produce a perturb-aware comparison:
  - TARC-4  (switch_cost=0.005, max_time_repeat=4, perturb=True, 5 seeds)
  - PPO-50Hz  (base_dt_divisor=1,   perturb=True, 5 seeds)
  - PPO-25Hz  (base_dt_divisor=2,   perturb=True, 5 seeds)
  - PPO-20Hz  (base_dt_divisor=2.5, perturb=True, 5 seeds)
  - PPO-12.5Hz (base_dt_divisor=4,  perturb=True, 5 seeds)

All runs log to project Go1_PerturbTrain.
After training, run eval_launcher.py / eval_baseline_launcher.py against
Go1_PerturbTrain with --force_perturb 1 to get eval results.

Usage (on Euler login node):
    cd ~/TARC/training/quadruped
    python launcher_perturb_retrain.py
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import training_go1
from training.euler_util import (generate_run_commands, generate_base_command,
                                  dict_permutations, available_gpus)

PROJECT = 'Go1_PerturbTrain'

tarc_config = {
    'env_name':             ['Go1JoystickFlatTerrain'],
    'backend':              ['generalized'],
    'project_name':         [PROJECT],
    'seed':                 list(range(5)),
    'switch_cost_wrapper':  [1],
    'switch_cost':          [0.005],
    'max_time_repeat':      [4, 5],
    'min_time_repeat':      [1],
    'time_as_part_of_state':[1],
    'num_final_evals':      [1],
    'perturb':              [1],
    'use_auto_lambda':      [0],
}

ppo_config = {
    'env_name':             ['Go1JoystickFlatTerrain'],
    'backend':              ['generalized'],
    'project_name':         [PROJECT],
    'seed':                 list(range(5)),
    'switch_cost_wrapper':  [0],
    'base_dt_divisor':      [1, 2, 2.5, 4],   # 50 Hz, 25 Hz, 20 Hz, 12.5 Hz
    'perturb':              [1],
    'num_final_evals':      [1],
}


def main():
    command_list = []

    for flags in dict_permutations(tarc_config):
        command_list.append(generate_base_command(training_go1, flags=flags))

    for flags in dict_permutations(ppo_config):
        command_list.append(generate_base_command(training_go1, flags=flags))

    n_tarc = len(list(dict_permutations(tarc_config)))
    n_ppo  = len(list(dict_permutations(ppo_config)))
    print(f'Jobs: {n_tarc} TARC-4  +  {n_ppo} PPO baselines  =  {n_tarc + n_ppo} total')

    generate_run_commands(command_list,
                          num_cpus=1,
                          num_gpus=1,
                          gpu=available_gpus[3],   # RTX 4090
                          mode='euler',
                          duration='03:59:00',
                          prompt=True,
                          mem=32000)


if __name__ == '__main__':
    main()
