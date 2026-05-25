"""Batch rollout evaluation for already-trained policies.

For each policy_params_<run_id>.pkl in --policies_dir, fetches the run's
hyperparameter config from W&B by run ID, then calls experiment() with
policy_path set so training is skipped and only the rollout runs.

Results are logged to a new run in <train_project>_Eval, named after the
original training run ID for traceability.

Usage:
    python eval_policies.py \
        --policies_dir /path/to/Policies \
        --train_project Go1_AutoLambda
"""

import os
import sys
import glob
import argparse
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import wandb
import training_go1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--policies_dir', type=str, required=True,
                        help='Directory containing policy_params_*.pkl files')
    parser.add_argument('--train_project', type=str, default='Go1_AutoLambda',
                        help='W&B project the policies were trained in')
    parser.add_argument('--entity', type=str, default='arnavsukhija-eth-zuric')
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

        print(f'Config: K_ratio={cfg.get("K_ratio")}, max_time_repeat={cfg.get("max_time_repeat")}, '
              f'seed={cfg.get("seed")}')

        try:
            training_go1.experiment(
                env_name=cfg.get('env_name', 'Go1JoystickFlatTerrain'),
                backend=cfg.get('backend', 'generalized'),
                project_name=args.train_project,
                seed=cfg.get('seed', 0),
                num_eval_envs=128,
                switch_cost_wrapper=bool(cfg.get('switch_cost_wrapper', True)),
                switch_cost=cfg.get('switch_cost', 0.0),
                max_time_repeat=int(cfg.get('max_time_repeat', 10)),
                min_time_repeat=int(cfg.get('min_time_repeat', 1)),
                time_as_part_of_state=bool(cfg.get('time_as_part_of_state', True)),
                num_final_evals=int(cfg.get('num_final_evals', 1)),
                perturb=bool(cfg.get('perturb', False)),
                base_dt_divisor=int(cfg.get('base_dt_divisor', 1)),
                use_auto_lambda=bool(cfg.get('use_auto_lambda', True)),
                K_ratio=float(cfg.get('K_ratio', 0.3)),
                alpha_lambda=float(cfg.get('alpha_lambda', 0.05)),
                lambda_init=float(cfg.get('lambda_init', 0.005)),
                lambda_min=float(cfg.get('lambda_min', 0.0)),
                lambda_max=float(cfg.get('lambda_max', 1.0)),
                policy_path=pkl_path,
                wandb_run_id=run_id,
            )
        except Exception:
            print(f'Error during rollout for {run_id}:')
            traceback.print_exc()
            # Ensure wandb run is closed even on error
            try:
                wandb.finish()
            except Exception:
                pass
            continue


if __name__ == '__main__':
    main()
