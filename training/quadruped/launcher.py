import training
from training.euler_util import generate_run_commands, generate_base_command, dict_permutations, available_gpus

################################################
#################### Quadruped ####################
################################################
go1_switch_cost = {'env_name': ['Go1JoystickFlatTerrain', ],
                     'backend': ['generalized', ],
                     'project_name': ["TaCoSGo1JoystickFlatTerrain_returnLastModel"],
                     'seed': list(range(5)),
                     'switch_cost': [0.005],
                     'max_time_repeat': [3,4,5,10],
                     'min_time_repeat': [1,],
                     'time_as_part_of_state': [1, ],
                     'num_final_evals': [1, ],
                      'perturb' : [0, ],
                     }

go1_no_switch_cost_ppo = {'env_name': ['Go1JoystickFlatTerrain', ],
                     'backend': ['generalized', ],
                     'project_name': ["PPOGo1JoystickFlatTerrain_returnLastModel"],
                     'seed': list(range(5)),
                     'switch_cost': [0, ],
                     'time_as_part_of_state': [0, ],
                     'num_final_evals': [1, ],
                     'perturb' : [0, ],
                     'switch_cost_wrapper': [0, ], # normal PPO (without switch cost wrapping)
                     }


def main():
    command_list = []
    flags_combinations = dict_permutations(go1_switch_cost)
    for flags in flags_combinations:
        cmd = generate_base_command(training, flags=flags)
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
