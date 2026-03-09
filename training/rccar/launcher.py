import os
# Temporarily set JAX to use CPU during launcher initialization on the login node 
# to prevent CUDA_ERROR_NO_DEVICE crashes.
os.environ["JAX_PLATFORMS"] = "cpu"

import exp
from training.euler_util import generate_run_commands, generate_base_command, dict_permutations, available_gpus

# Remove the CPU override so that the Slurm jobs inherit the default behavior and use the GPU!
os.environ.pop("JAX_PLATFORMS", None)

################################################
#################### RC Car ####################
################################################

rccar_switch_cost = {'env_name': ['rccar', ],
                     'backend': ['generalized', ],
                     'project_name': ["TARC_RCCar"],
                     'num_timesteps': [75_000_000, ],
                     'episode_steps': [200, ],
                     'base_discount_factor': [0.9],
                     'seed': list(range(5)),
                     'num_envs': [2048],
                     'num_eval_envs': [32],
                     'entropy_cost': [1e-2],
                     'unroll_length': [10],
                     'num_minibatches': [32],
                     'num_updates_per_batch': [4],
                     'batch_size': [1024],
                     'networks': [0, ],
                     'reward_scaling': [1.0, ],
                     'switch_cost': [0.1, ],
                     'max_time_repeat': [2,3,4,5,10],
                     'time_as_part_of_state': [1, ],
                     'num_final_evals': [10, ],
                     'domain_randomization': [1, ],
                     'sample_init_pos': [1,],
                     'action_delay': [2.0]
                     }

rccar_no_switch_cost_ppo = {'env_name': ['rccar', ],
                     'backend': ['generalized', ],
                     'project_name': ["PPO_RCCar_lowFrequency"],
                     'num_timesteps': [75_000_000, ], #from normal ppo training
                     'episode_steps': [200, ],
                     'base_discount_factor': [0.9],
                     'seed': list(range(5)),
                     'num_envs': [2048],
                     'num_eval_envs': [32],
                     'entropy_cost': [1e-2],
                     'unroll_length': [10],
                     'num_minibatches': [32],
                     'num_updates_per_batch': [4],
                     'batch_size': [1024],
                     'networks': [0, ],
                     'reward_scaling': [1.0, ],
                     'switch_cost': [0.1, ],
                     'max_time_repeat': [10],
                     'time_as_part_of_state': [0, ],
                     'num_final_evals': [10, ],
                    'switch_cost_wrapper': [0, ], # normal PPO (without switch cost wrapping)
                    'domain_randomization': [1,],
                    'sample_init_pos': [1,],
                    'action_delay': [2.0],
                    'base_dt_divisor': [2]
                     }


def main():
    command_list = []
    flags_combinations = dict_permutations(rccar_no_switch_cost_ppo)
    for flags in flags_combinations:
        cmd = generate_base_command(exp, flags=flags)
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
