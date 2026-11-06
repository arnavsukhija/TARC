"""Euler launcher: 10-rollout eval for non-TARC baseline policies (Go1).

Walks --policies_dir, fetches each run's W&B config from --train_project,
keeps only runs that match the baseline filter (switch_cost_wrapper == False
AND base_dt_divisor == --base_dt_divisor if specified), and submits one
SBATCH job per matching policy. Each job runs training_go1.py with
--policy_path set, which triggers the eval-mode branch that does 10
random-command rollouts logged under Results_{i}_random/*.

Usage (on Euler login node):
    cd ~/TARC/training/quadruped
    # Eval all PPO baselines:
    python eval_baseline_launcher.py \
        --policies_dir /cluster/scratch/asukhija/Policies \
        --train_project Go1_Stability_Rebuttal \
        --eval_project Go1_Stability_Rebuttal_Eval
    # Eval only PPO-20Hz (divisor=2.5):
    python eval_baseline_launcher.py \
        --policies_dir /cluster/scratch/asukhija/Policies \
        --train_project Go1_Stability_Rebuttal \
        --eval_project Go1_Stability_Rebuttal_Eval \
        --base_dt_divisor 2.5
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
    parser.add_argument('--train_project', type=str, default='Go1_Stability_Rebuttal',
                        help='W&B project the baseline policies were trained in')
    parser.add_argument('--baseline_steps', type=int, default=None,
                        help='Only eval runs with this num_timesteps (omit to accept all)')
    parser.add_argument('--base_dt_divisor', type=float, default=None,
                        help='Only eval runs with this base_dt_divisor, e.g. 2.5 for PPO-20Hz '
                             '(omit to accept all non-TARC runs)')
    parser.add_argument('--eval_project', type=str, default=None,
                        help='W&B project to log eval results into (defaults to train_project)')
    parser.add_argument('--duration', type=str, default='00:30:00')
    parser.add_argument('--gpu', type=int, default=3,
                        help='GPU index from available_gpus (default: 3 = RTX 4090)')
    parser.add_argument('--force_perturb', type=int, default=None,
                        help='Override perturb: 1=force on, 0=force off, omit=use training config')
    parser.add_argument('--dynamic_commands', type=int, default=0,
                        help='If 1, resample command every --command_change_steps inner steps')
    parser.add_argument('--command_change_steps', type=int, default=250)
    parser.add_argument('--perturb_velocity_kick_max', type=float, default=None,
                        help='Override max velocity kick magnitude (m/s); default env value is 3.0.')
    parser.add_argument('--perturb_kick_wait_min', type=float, default=None,
                        help='Override min wait time between kicks (s); default env value is 1.0.')
    parser.add_argument('--perturb_kick_wait_max', type=float, default=None,
                        help='Override max wait time between kicks (s); default env value is 3.0.')
    args = parser.parse_args()

    pkl_files = sorted(glob.glob(os.path.join(os.path.expanduser(args.policies_dir),
                                              'policy_params_*.pkl')))
    if not pkl_files:
        print(f'No policy_params_*.pkl files found in {args.policies_dir}')
        return
    filters = ['switch_cost_wrapper=False']
    if args.baseline_steps is not None:
        filters.append(f'num_timesteps={args.baseline_steps}')
    if args.base_dt_divisor is not None:
        filters.append(f'base_dt_divisor={args.base_dt_divisor}')
    print(f'Found {len(pkl_files)} candidate policies. Filtering by config '
          f'({", ".join(filters)}) ...')

    api = wandb.Api()
    command_list = []
    kept, skipped_missing, skipped_filter = 0, 0, 0

    for pkl_path in pkl_files:
        run_id = os.path.basename(pkl_path).removeprefix('policy_params_').removesuffix('.pkl')

        try:
            run = api.run(f'{ENTITY}/{args.train_project}/{run_id}')
            cfg = run.config
        except Exception as e:
            print(f'  [skip] {run_id}: could not fetch W&B config ({e})')
            skipped_missing += 1
            continue

        is_tarc = bool(cfg.get('switch_cost_wrapper', False))
        nts = int(cfg.get('num_timesteps', 0))
        cfg_div = float(cfg.get('base_dt_divisor', 1))
        skip_reason = None
        if is_tarc:
            skip_reason = 'switch_cost_wrapper=True'
        elif args.baseline_steps is not None and nts != args.baseline_steps:
            skip_reason = f'num_timesteps={nts} (want {args.baseline_steps})'
        elif args.base_dt_divisor is not None and abs(cfg_div - args.base_dt_divisor) > 1e-6:
            skip_reason = f'base_dt_divisor={cfg_div} (want {args.base_dt_divisor})'
        if skip_reason:
            print(f'  [skip] {run_id}: {skip_reason}')
            skipped_filter += 1
            continue

        print(f'  [keep] {run_id}: base_dt_divisor={cfg_div}, num_timesteps={nts}')

        flags = {
            'env_name':             cfg.get('env_name', 'Go1JoystickFlatTerrain'),
            'backend':              cfg.get('backend', 'generalized'),
            'project_name':         args.train_project,
            'seed':                 cfg.get('seed', 0),
            'num_eval_envs':        128,
            'switch_cost_wrapper':  0,
            'switch_cost':          float(cfg.get('switch_cost', 0.0)),
            'max_time_repeat':      int(cfg.get('max_time_repeat', 10)),
            'min_time_repeat':      int(cfg.get('min_time_repeat', 1)),
            'time_as_part_of_state': int(bool(cfg.get('time_as_part_of_state', True))),
            'num_final_evals':      int(cfg.get('num_final_evals', 1)),
            'perturb':              args.force_perturb if args.force_perturb is not None else int(bool(cfg.get('perturb', False))),
            'base_dt_divisor':      float(cfg.get('base_dt_divisor', 1)),
            'num_timesteps':        nts,
            'use_auto_lambda':      0,
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
        kept += 1

    print(f'\nSummary: kept={kept}, skipped_missing_config={skipped_missing}, '
          f'skipped_by_filter={skipped_filter}')
    if not command_list:
        print('No baseline policies to evaluate.')
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
