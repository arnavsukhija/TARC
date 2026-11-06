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
    parser.add_argument('--max_time_repeat', type=int, default=None,
                        help='Only eval runs with this max_time_repeat (omit to eval all)')
    parser.add_argument('--switch_cost', type=float, default=None,
                        help='Only eval runs with this switch_cost (omit to eval all)')
    parser.add_argument('--eval_project', type=str, default=None)
    parser.add_argument('--base_dt_divisor', type=float, default=None,
                        help='Only eval runs with this base_dt_divisor (omit to eval all)')
    parser.add_argument('--switch_cost_wrapper', type=int, default=None,
                        help='Only eval runs with this switch_cost_wrapper value (0 or 1)')
    parser.add_argument('--force_perturb', type=int, default=None,
                        help='Override perturb setting: 1=force on, 0=force off, omit=use training config')
    parser.add_argument('--dynamic_commands', type=int, default=0,
                        help='If 1, resample command every --command_change_steps inner steps during eval')
    parser.add_argument('--command_change_steps', type=int, default=250)
    parser.add_argument('--perturb_velocity_kick_max', type=float, default=None,
                        help='Override max velocity kick magnitude (m/s); default env value is 3.0.')
    parser.add_argument('--perturb_kick_wait_min', type=float, default=None,
                        help='Override min wait time between kicks (s); default env value is 1.0.')
    parser.add_argument('--perturb_kick_wait_max', type=float, default=None,
                        help='Override max wait time between kicks (s); default env value is 3.0.')
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

        cfg_mtr = cfg.get('max_time_repeat')
        cfg_sc  = cfg.get('switch_cost')
        if args.max_time_repeat is not None and cfg_mtr != args.max_time_repeat:
            print(f'  {run_id}: skipping (max_time_repeat={cfg_mtr}, want {args.max_time_repeat})')
            continue
        if args.switch_cost is not None and not (cfg_sc is not None and abs(float(cfg_sc) - args.switch_cost) < 1e-9):
            print(f'  {run_id}: skipping (switch_cost={cfg_sc}, want {args.switch_cost})')
            continue
        cfg_div = float(cfg.get('base_dt_divisor', 1))
        if args.base_dt_divisor is not None and abs(cfg_div - args.base_dt_divisor) > 1e-6:
            print(f'  {run_id}: skipping (base_dt_divisor={cfg_div}, want {args.base_dt_divisor})')
            continue
        cfg_scw = int(bool(cfg.get('switch_cost_wrapper', False)))
        if args.switch_cost_wrapper is not None and cfg_scw != args.switch_cost_wrapper:
            print(f'  {run_id}: skipping (switch_cost_wrapper={cfg_scw}, want {args.switch_cost_wrapper})')
            continue
        print(f'  {run_id}: K_ratio={cfg.get("K_ratio")}, max_time_repeat={cfg_mtr}, switch_cost={cfg_sc}, base_dt_divisor={cfg_div} -> QUEUED')

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
            'perturb':              args.force_perturb if args.force_perturb is not None else int(bool(cfg.get('perturb', False))),
            'base_dt_divisor':      float(cfg.get('base_dt_divisor', 1)),
            'use_auto_lambda':      int(bool(cfg.get('use_auto_lambda', False))),
            'K_ratio':              float(cfg.get('K_ratio', 0.3)),
            'alpha_lambda':         float(cfg.get('alpha_lambda', 0.05)),
            'lambda_init':          float(cfg.get('lambda_init', 0.005)),
            'lambda_min':           float(cfg.get('lambda_min', 0.0)),
            'lambda_max':           float(cfg.get('lambda_max', 1.0)),
            'policy_path':          pkl_path,
            'wandb_run_id':         run_id,
        }
        if args.eval_project is not None:
            flags['eval_project'] = args.eval_project
        if args.dynamic_commands:
            flags['dynamic_commands'] = 1
            flags['command_change_steps'] = args.command_change_steps
        if args.perturb_velocity_kick_max is not None:
            flags['perturb_velocity_kick_max'] = args.perturb_velocity_kick_max
        if args.perturb_kick_wait_min is not None:
            flags['perturb_kick_wait_min'] = args.perturb_kick_wait_min
        if args.perturb_kick_wait_max is not None:
            flags['perturb_kick_wait_max'] = args.perturb_kick_wait_max
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
