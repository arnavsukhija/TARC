# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TARC (Time-Adaptive Robotic Control) is a JAX/Brax-based reinforcement learning framework that learns policies with variable action timing. The key idea: instead of applying actions at fixed time intervals, the policy outputs both the control action AND how long to hold it (via a switching cost formulation). This is implemented through environment wrappers that augment the action space with a timing dimension.

## Setup

```bash
conda activate tarc_jax
pip install -e .
```

Environment variables needed for GPU training (set in `setup_env.sh`):
```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL="egl"
export PYTHONPATH="$HOME/TARC:$PYTHONPATH"
```

The `setup_env.sh` script is for ETH Euler cluster (Slurm/SBATCH) initialization — not needed for local runs.

## Running Experiments

**RC Car simulation (single run):**
```bash
python training/rccar/exp.py --switch_cost_wrapper 1 --switch_cost 0.1 --max_time_repeat 5
```

**Go1 quadruped (single run):**
```bash
python training/quadruped/training_go1.py --switch_cost_wrapper 1 --switch_cost 0.1
```

**Launching hyperparameter sweeps on ETH Euler cluster:**
```bash
cd training/rccar && python launcher.py   # prompts before submitting Slurm jobs
cd training/quadruped && python launcher.py
```

The launchers call `euler_util.generate_run_commands()` which generates SBATCH commands. The launcher sets `JAX_PLATFORMS=cpu` during initialization on the login node to avoid CUDA errors, then unsets it before submitting.

All experiments log to Weights & Biases (entity: `asukhija`). Policies and trajectories are saved via `cloudpickle` both locally and as W&B artifacts.

## Architecture

### Core Concept: Switch Cost Wrappers

The central contribution is in `wrappers/`. The policy outputs `action = [u, pseudo_time]` where `pseudo_time ∈ [-1, 1]` is mapped to `[t_min, t_max]`. The wrapper integrates the underlying environment forward for the chosen duration, accumulating discounted rewards, then subtracts a switching cost.

Two wrapper variants:
- **`wrappers/switching_cost.py`** (`SwitchCostWrapper`): tracks time as `time_to_go` (decreasing), appends to obs
- **`wrappers/ih_switching_cost.py`** (`IHSwitchCostWrapper`): tracks time as elapsed time (increasing); the `time_as_part_of_state=True` flag appends elapsed time to obs; uses `while_loop` for JAX-compatible dynamic number of steps
- **`wrappers/ih_switching_cost_mjx.py`**: MJX variant for quadruped (MuJoCo Playground)
- **`wrappers/ih_switching_cost_gym.py`**: gym-API variant for hardware deployment

Each wrapper also has a `simulation_step()` method (Python while-loop, not JIT-able) used during evaluation to capture intermediate states.

### Environments

- **RC Car sim** (`envs/rccar/sim/rccar.py`): Brax `PipelineEnv` with a 6D state `[x, y, θ, vx, vy, ω]`. `dt_divisor` controls integration frequency (e.g., `dt_divisor=2` → 30 Hz base). Supports domain randomization and initial position sampling.
- **RC Car hardware** (`envs/rccar/hardware/car_env_hardware.py`): gym `Env` using `carl` (Windows-only C++ library) for physical robot control via motion capture.
- **Quadruped** (`training/quadruped/training_go1.py`): Uses `mujoco_playground` registry for the Go1 robot; wrapped with `IHSwitchCostWrapper` from the MJX variant.

### Optimizers (PPO)

Two parallel PPO implementations with nearly identical APIs:

- **`optimizer/ppo_mbpo/`**: Used for RC car. Wraps Brax environments via `brax.envs.wrappers.training.wrap`.
- **`optimizer/ppo/`**: Used for quadruped. Uses `mujoco_playground._src.wrapper.wrap_for_brax_training`. Contains a patch for `BraxDomainRandomizationVmapWrapper.v_env_fn`.

Both expose a `PPO` class with `run_training(key, progress_fn)` → `(policy_params, metrics)` and `make_policy(params, deterministic)`.

Key TARC-specific PPO parameters:
- `non_equidistant_time=True`: enables non-uniform time discounting in the loss
- `continuous_discounting`: computed from discrete discount factor via `utils/discounting.py`
- `min_time_between_switches`, `max_time_between_switches`: passed through to loss computation

### Discounting

`utils/discounting.py` converts between discrete and continuous discount factors:
- `discrete_to_continuous_discounting(γ, dt) = log(1/γ) / dt`
- `continuous_to_discrete_discounting(λ, dt) = exp(-λ·dt)`

This is necessary because with variable action durations, the discount applied per action depends on its duration.

### Cluster Utilities (`training/euler_util.py`)

`generate_run_commands(command_list, num_cpus, num_gpus, gpu, mode, duration, mem, prompt)` generates and optionally submits Slurm `sbatch` commands. GPU types are indexed in `available_gpus` (e.g., index 3 = RTX 4090). The `dict_permutations()` function creates the Cartesian product of hyperparameter dicts for sweep configs.

### Action Delay (`wrappers/ActionDelayWrapper.py`)

Augments observation with a rolling action buffer to implement fractional action delays. The delayed action is a linear interpolation between the two oldest buffered actions, controlled by `interp_weights`. The `action_delay` parameter is in units of timesteps.
