# TARC: Time-Adaptive Robotic Control

[![Website](https://img.shields.io/badge/Website-blue)](https://arnavsukhija.github.io/projects/tarc/)
[![arXiv](https://img.shields.io/badge/arXiv-red)](https://arxiv.org/abs/2510.23176)

Official implementation of **TARC: Time-Adaptive Robotic Control** (Sukhija et al., CoRL 2026)

Presented as a spotlight paper at Conference on Robot Learning, 2026 in Austin, TX.

Standard RL policies act at a fixed control frequency. TARC learns policies that output both the
control action **and how long to hold it**. The policy outputs `[u, pseudo_time]`, where
`pseudo_time ∈ [-1, 1]` is mapped to `[t_min, t_max]`. An environment wrapper integrates the system
for that duration and subtracts a switching cost, so the policy learns to act only when it matters.

- [**Project page**](https://arnavsukhija.github.io/projects/tarc/)
- [**Paper**](https://arxiv.org/abs/2510.23176)

## Repository layout

| Path | Contents |
|---|---|
| `wrappers/` | Switch-cost wrappers (the core contribution): Brax, MJX, and gym (hardware) variants, plus the action-delay wrapper |
| `optimizer/` | PPO implementations with non-equidistant time discounting (`ppo_mbpo/` for RC car, `ppo/` for quadruped) |
| `envs/` | RC car (simulation and hardware) and humanoid environments |
| `training/` | Training, evaluation and Slurm sweep launchers for `rccar/`, `quadruped/`, `humanoid/` |
| `plotting/` | Scripts that generate the paper figures |
| `results/` | Data used by the plots: Go1 sim2real CSVs and RC car hardware trajectories (`results/rccar/trajectories/`) |
| `utils/` | Discounting conversions and helpers |
| `dsrl_pi0/` | Git submodule: [arnavsukhija/dsrl_pi0](https://github.com/arnavsukhija/dsrl_pi0), TARC applied to DSRL with π₀ |

## Installation

```bash
git clone --recurse-submodules git@github.com:arnavsukhija/TARC.git
cd TARC

conda create -n tarc_jax python=3.11
conda activate tarc_jax
pip install -e .
```

If you cloned without `--recurse-submodules`: `git submodule update --init --recursive`.

For GPU training:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL="egl"
export PYTHONPATH="$PWD:$PYTHONPATH"
```

`setup_env.sh` does this for the ETH Euler cluster (Slurm). You don't need it for local runs.
Install the CUDA build of JAX (`pip install -U "jax[cuda12]"`) for GPU use.
Experiments log to [Weights & Biases](https://wandb.ai). Run `wandb login` first.

## Training

**RC car (simulation):**

```bash
python training/rccar/exp.py --switch_cost_wrapper 1 --switch_cost 0.1 --max_time_repeat 5
```

**Unitree Go1 (MuJoCo Playground):**

```bash
python training/quadruped/training_go1.py --switch_cost_wrapper 1 --switch_cost 0.1
```

**Humanoid (G1):** `python training/humanoid/training_g1.py`

Set `--switch_cost_wrapper 0` to train the fixed-frequency PPO baseline.

**Hyperparameter sweeps (Slurm):** each task folder has a `launcher.py` that builds the sweep and
prompts before submitting jobs:

```bash
cd training/rccar && python launcher.py
cd training/quadruped && python launcher.py
```

Policies and trajectories are saved with `cloudpickle`, locally and as W&B artifacts.

**DSRL with π₀:** see [`dsrl_pi0/README.md`](dsrl_pi0/README.md).

## Evaluation and deployment

Evaluate trained policies with `training/<task>/eval_policies.py` (or `eval_launcher.py` on the cluster).
Evaluation uses `simulation_step()` on the wrappers, which records the intermediate states of each
held action.

**RC car hardware.** `envs/rccar/hardware/car_env_hardware.py` is a gym environment that drives the
physical car through motion capture using the `carl` library (Windows only). Wrap it with
`wrappers/ih_switching_cost_gym.py` and load a policy saved by training.
TODO: add a minimal deployment script and a link to the hardware setup.

**Go1 hardware.** Policies are trained with `wrappers/ih_switching_cost_mjx.py`. The hold duration
is chosen by the policy at run time.
TODO: describe the deployment pipeline.

## Reproducing the figures

Plot scripts live in `plotting/` and pull runs from W&B, so set the entity and project names in each script first.

## Citation

```bibtex
@misc{sukhija2026tarctimeadaptiveroboticcontrol,
      title={TARC: Time-Adaptive Robotic Control}, 
      author={Arnav Sukhija and Lenart Treven and Jin Cheng and Florian Dörfler and Stelian Coros and Andreas Krause},
      year={2026},
      eprint={2510.23176},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2510.23176}, 
}
```

## License

MIT, see [LICENSE](LICENSE).
