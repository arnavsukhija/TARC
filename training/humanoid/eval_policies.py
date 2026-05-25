"""Batch rollout evaluation for already-trained G1 policies.

For each policy_params_<run_id>.pkl in --policies_dir, fetches the run's
hyperparameter config from W&B by run ID, then calls experiment() with
policy_path set so training is skipped and only the rollout runs.

Results are logged to a new run in <train_project>_Eval, named after the
original training run ID for traceability.

Usage:
    python eval_policies.py \
        --policies_dir /path/to/Policies \
        --train_project G1_TARC
"""

import os
import sys
import glob
import argparse
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import wandb
import training_g1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policies_dir', type=str, required=True,
                        help='Directory containing policy_params_*.pkl files')
    parser.add_argument('--train_project', type=str, default='G1_TARC',
                        help='W&B project the policies were trained in')
    parser.add_argument('--entity', type=str, default='arnavsukhija-eth-zurich')
    args = parser.parse_args()

    pkl_files = sorted(glob.glob(os.path.join(args.policies_dir, 'policy_params_*.pkl')))
    if not pkl_files:
        print(f'No policy_params_*.pkl files found in {args.policies_dir}')
        return
    print(f'Found {len(pkl_files)} policies to evaluate.')

    api = wandb.Api()

    for pkl_path in pkl_files:
        run_id = os.path.basename(pkl_path).removeprefix('policy_params_').removesuffix('.pkl')
        print(f'\n{"="*60}')
        print(f'Evaluating run {run_id}  ({pkl_path})')

        try:
            run = api.run(f'{args.entity}/{args.train_project}/{run_id}')
            cfg = run.config
        except Exception as e:
            print(f'Could not fetch W&B config for {run_id}: {e}. Skipping.')
            continue

        print(f'Config: switch_cost={cfg.get("switch_cost")}, max_time_repeat={cfg.get("max_time_repeat")}, '
              f'seed={cfg.get("seed")}')

        try:
            training_g1.experiment(
                env_name=cfg.get('env_name', 'G1JoystickFlatTerrain'),
                backend=cfg.get('backend', 'generalized'),
                project_name=args.train_project,
                seed=cfg.get('seed', 0),
                num_eval_envs=128,
                switch_cost_wrapper=bool(cfg.get('switch_cost_wrapper', True)),
                switch_cost=float(cfg.get('switch_cost', 0.005)),
                max_time_repeat=int(cfg.get('max_time_repeat', 4)),
                min_time_repeat=int(cfg.get('min_time_repeat', 1)),
                time_as_part_of_state=bool(cfg.get('time_as_part_of_state', True)),
                num_final_evals=int(cfg.get('num_final_evals', 1)),
                perturb=bool(cfg.get('perturb', False)),
                base_dt_divisor=int(cfg.get('base_dt_divisor', 1)),
                policy_path=pkl_path,
                wandb_run_id=run_id,
            )
        except Exception:
            print(f'Error during rollout for {run_id}:')
            traceback.print_exc()
            try:
                wandb.finish()
            except Exception:
                pass
            continue


if __name__ == '__main__':
    main()
