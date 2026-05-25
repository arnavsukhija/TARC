"""Euler launcher: submit one SBATCH eval job per saved policy.

Reads policy_params_<run_id>.pkl files from --policies_dir, fetches each
run's hyperparameter config from W&B on the login node (no GPU needed),
then submits one sbatch job per policy that runs training_go1.py with
--policy_path and --wandb_run_id so results are logged back into the
original training run.

Usage (on Euler login node):
    cd ~/TARC/training/quadruped
    python eval_launcher.py \
        --policies_dir /cluster/scratch/asukhija/Policies \
        --train_project Go1_AutoLambda
"""

import os
import sys
import glob
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import wandb
import training_go1
from training.euler_util import generate_run_commands, generate_base_command, available_gpus

ENTITY = 'arnavsukhija-eth-zurich'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policies_dir', type=str, required=True,
                        help='Directory containing policy_params_*.pkl files')
    parser.add_argument('--train_project', type=str, default='Go1_AutoLambda',
                        help='W&B project the policies were trained in')
    parser.add_argument('--duration', type=str, default='00:30:00',
                        help='SBATCH wall-clock limit per job (eval is short)')
    parser.add_argument('--gpu', type=int, default=3,
                        help='GPU index from available_gpus (default: 3 = RTX 4090)')
    args = parser.parse_args()

    pkl_files = sorted(glob.glob(os.path.join(args.policies_dir, 'policy_params_*.pkl')))
    if not pkl_files:
        print(f'No policy_params_*.pkl files found in {args.policies_dir}')
        return
    print(f'Found {len(pkl_files)} policies.')

    api = wandb.Api()
    command_list = []

    for pkl_path in pkl_files:
        run_id = os.path.basename(pkl_path).removeprefix('policy_params_').removesuffix('.pkl')

        try:
            run = api.run(f'{ENTITY}/{args.train_project}/{run_id}')
            cfg = run.config
        except Exception as e:
            print(f'Could not fetch W&B config for {run_id}: {e}. Skipping.')
            continue

        print(f'  {run_id}: K_ratio={cfg.get("K_ratio")}, max_time_repeat={cfg.get("max_time_repeat")}')

        flags = {
            'env_name':             cfg.get('env_name', 'Go1JoystickFlatTerrain'),
            'backend':              cfg.get('backend', 'generalized'),
            'project_name':         args.train_project,
            'seed':                 cfg.get('seed', 0),
            'num_eval_envs':        128,
            'switch_cost_wrapper':  int(bool(cfg.get('switch_cost_wrapper', True))),
            'switch_cost':          float(cfg.get('switch_cost', 0.0)),
            'max_time_repeat':      int(cfg.get('max_time_repeat', 10)),
            'min_time_repeat':      int(cfg.get('min_time_repeat', 1)),
            'time_as_part_of_state': int(bool(cfg.get('time_as_part_of_state', True))),
            'num_final_evals':      int(cfg.get('num_final_evals', 1)),
            'perturb':              int(bool(cfg.get('perturb', False))),
            'base_dt_divisor':      int(cfg.get('base_dt_divisor', 1)),
            'use_auto_lambda':      int(bool(cfg.get('use_auto_lambda', True))),
            'K_ratio':              float(cfg.get('K_ratio', 0.3)),
            'alpha_lambda':         float(cfg.get('alpha_lambda', 0.05)),
            'lambda_init':          float(cfg.get('lambda_init', 0.005)),
            'lambda_min':           float(cfg.get('lambda_min', 0.0)),
            'lambda_max':           float(cfg.get('lambda_max', 1.0)),
            'policy_path':          pkl_path,
            'wandb_run_id':         run_id,
        }
        command_list.append(generate_base_command(training_go1, flags=flags))

    if not command_list:
        print('No valid commands generated.')
        return

    generate_run_commands(command_list,
                          num_cpus=1,
                          num_gpus=1,
                          gpu=available_gpus[args.gpu],
                          mode='euler',
                          duration=args.duration,
                          prompt=True,
                          mem=32000)


if __name__ == '__main__':
    main()
