import argparse
import datetime
import functools
import cloudpickle
from datetime import datetime
import os

import mediapy
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import wandb

from jax.nn import swish

from optimizer.ppo.ppo_brax_env import PPO
from wrappers.ih_switching_cost_mjx import ConstantSwitchCost, IHSwitchCostWrapper

from mujoco_playground import registry
from mujoco_playground.config import locomotion_params
from envs.humanoid.g1_custom import G1JoystickCustom

import contextlib
from mujoco_playground._src import wrapper as mj_wrapper
from mujoco import mjx

def patched_v_env_fn(self, mjx_model: mjx.Model):
    env = self.env
    unwrapped = env.unwrapped
    old_mjx_model = unwrapped._mjx_model
    try:
        unwrapped._mjx_model = mjx_model
        yield env
    finally:
        unwrapped._mjx_model = old_mjx_model

mj_wrapper.BraxDomainRandomizationVmapWrapper.v_env_fn = contextlib.contextmanager(patched_v_env_fn)


def _load_g1_env(env_name: str, env_cfg):
    task = "flat_terrain" if "Flat" in env_name else "rough_terrain"
    return G1JoystickCustom(task=task, config=env_cfg)


def _apply_reward_overrides(env_cfg):
    env_cfg.reward_config.scales.torques = -1e-4
    env_cfg.reward_config.scales.action_rate = -1e-2
    env_cfg.reward_config.scales.energy = -1e-4
    env_cfg.reward_config.scales.dof_acc = -2.5e-7
    return env_cfg


from jax import config

config.update("jax_debug_nans", True)

ENTITY = 'asukhija'

# G1 reads body-frame sensors per body (e.g. "pelvis", "torso") — pass this
# everywhere we query velocities/gyro on the eval env.
G1_SENSOR_FRAME = "pelvis"

# Video fps to write (matches ctrl_dt=0.02 → 50Hz).
VIDEO_FPS = 50


def save_video_mp4(frames: list, path: str, fps: int = VIDEO_FPS) -> None:
    """Write a list of HxWx3 uint8 numpy frames to an mp4 file."""
    mediapy.write_video(path, [np.asarray(f, dtype=np.uint8) for f in frames], fps=fps)


def rollout_video_tarc(eval_env, jit_reset, jit_inference_fn, episode_length,
                        rng, modify_scene_fns_out: list) -> list:
    """Run one episode using simulation_step (Python loop, not JIT) to collect
    inner-env states at ctrl_dt resolution for smooth video rendering.

    Returns the list of inner mjx_env.State objects (each at ctrl_dt spacing).
    Also appends one modify_scene_fn per inner state into modify_scene_fns_out.
    """
    from mujoco_playground._src.gait import draw_joystick_command

    state = jit_reset(rng)
    env_steps = 0
    inner_states = []
    cmd_amp = jnp.array(eval_env.env._config.command_config.a)

    while env_steps < episode_length:
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        next_state, inner_part = eval_env.simulation_step(state, ctrl)
        # inner_part is a stacked batch of states at ctrl_dt spacing
        n_inner = inner_part.reward.shape[0]
        for k in range(n_inner):
            s_k = jax.tree_util.tree_map(lambda x: x[k], inner_part)
            inner_states.append(s_k)
            xyz = np.array(s_k.data.xpos[eval_env.env._torso_body_id])
            xyz += np.array([0, 0, 0.3])
            x_axis = s_k.data.xmat[eval_env.env._torso_body_id, 0]
            yaw = -np.arctan2(x_axis[1], x_axis[0])
            modify_scene_fns_out.append(
                functools.partial(
                    draw_joystick_command,
                    cmd=s_k.info["command"],
                    xyz=xyz,
                    theta=yaw,
                    scl=abs(float(s_k.info["command"][0])) / float(cmd_amp[0]),
                )
            )
        predicted_steps = int(eval_env.compute_steps(pseudo_time=ctrl[-1]))
        env_steps += predicted_steps
        state = next_state
        if bool(state.done):
            break

    return inner_states


def save_policy(policy_params):
    if wandb.run is None:
        raise RuntimeError("wandb.run is not initialized. Ensure wandb.init() is called before logging artifacts.")

    directory = os.path.join(os.getcwd(), 'Policies')
    if not os.path.exists(directory):
        os.makedirs(directory)

    policy_path = os.path.join(directory, f"policy_params_{wandb.run.id}.pkl")

    try:
        with open(policy_path, "wb") as f:
            cloudpickle.dump(policy_params, f)

        try:
            with open(policy_path, "rb") as f:
                cloudpickle.load(f)
            print("Successfully loaded policy from file for verification.")
        except Exception as e:
            print(f"Error loading policy file for verification: {e}")
            return

        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"File not found: {policy_path}")

        wandb.save(policy_path, wandb.run.dir)
        print(f"Successfully saved and uploaded {policy_path} to Weights & Biases.")

    except Exception as e:
        print(f"An error occurred during policy upload: {e}")
    print("Policy saved to wandb!")


def experiment(env_name: str = 'G1JoystickFlatTerrain',
               backend: str = 'generalized',
               project_name: str = 'G1_TARC',
               seed: int = 0,
               num_eval_envs: int = 128,
               switch_cost_wrapper: bool = False,
               switch_cost: float = 0.005,
               max_time_repeat: int = 4,
               min_time_repeat: int = 1,
               time_as_part_of_state: bool = True,
               num_final_evals: int = 1,
               perturb: bool = False,
               base_dt_divisor: int = 1,
               policy_path: str = None,
               wandb_run_id: str = None,
               ):

    env_cfg = registry.get_default_config(env_name)
    if isinstance(env_cfg, dict):
        env_cfg['impl'] = 'jax'
    elif hasattr(env_cfg, 'impl'):
        env_cfg.impl = 'jax'
    if hasattr(env_cfg, 'sim_config'):
        if isinstance(env_cfg.sim_config, dict):
            env_cfg.sim_config['impl'] = 'jax'
        elif hasattr(env_cfg.sim_config, 'impl'):
            env_cfg.sim_config.impl = 'jax'

    if perturb and hasattr(env_cfg, 'pert_config'):
        env_cfg.pert_config.enable = True

    _apply_reward_overrides(env_cfg)

    if base_dt_divisor > 1:
        env_cfg.ctrl_dt *= base_dt_divisor
        print(f"Lowering control frequency: new ctrl_dt = {env_cfg.ctrl_dt} ({1.0/env_cfg.ctrl_dt:.2f} Hz)")

    control_frequency_hz = 1.0 / env_cfg.ctrl_dt
    g1_env = _load_g1_env(env_name, env_cfg)
    ppo_params = locomotion_params.brax_ppo_config(env_name)
    ppo_config = dict(ppo_params)
    action_repeat = ppo_config['action_repeat']
    batch_size = ppo_config['batch_size']
    discount_factor = ppo_config['discounting']
    entropy_cost = ppo_config['entropy_cost']
    episode_length = ppo_config['episode_length']
    if base_dt_divisor > 1:
        episode_length = episode_length // base_dt_divisor
        print(f"Adjusted episode_length = {episode_length} (same physical time: {episode_length * env_cfg.ctrl_dt:.1f}s)")
    learning_rate = ppo_config['learning_rate']
    max_grad_norm = ppo_config['max_grad_norm']
    policy_hidden_layer_sizes = ppo_config['network_factory']['policy_hidden_layer_sizes']
    critic_hidden_layer_sizes = ppo_config['network_factory']['value_hidden_layer_sizes']
    value_obs_key = ppo_config['network_factory']['value_obs_key']
    policy_obs_key = ppo_config['network_factory']['policy_obs_key']
    normalize_observations = ppo_config['normalize_observations']
    num_envs = ppo_config['num_envs']
    num_evals = ppo_config['num_evals']
    num_minibatches = ppo_config['num_minibatches']
    num_resets_per_eval = ppo_config['num_resets_per_eval']
    num_timesteps = 1_500_000_000 if switch_cost_wrapper else 3_500_000_000
    num_updates_per_batch = ppo_config['num_updates_per_batch']
    reward_scaling = ppo_config['reward_scaling']
    unroll_length = ppo_config['unroll_length']
    sim_dt = env_cfg['sim_dt']
    ctrl_dt = env_cfg['ctrl_dt']

    randomization_fn = registry.get_domain_randomizer(env_name)

    if switch_cost_wrapper:
        env = IHSwitchCostWrapper(env=g1_env,
                                  episode_steps=episode_length,
                                  min_time_between_switches=min_time_repeat,
                                  max_time_between_switches=max_time_repeat,
                                  switch_cost=ConstantSwitchCost(value=jnp.array(switch_cost)),
                                  discounting=discount_factor,
                                  time_as_part_of_state=time_as_part_of_state,
                                  sim_dt=sim_dt,
                                  )
        eval_env = IHSwitchCostWrapper(env=_load_g1_env(env_name, env_cfg),
                                  episode_steps=episode_length,
                                  min_time_between_switches=min_time_repeat,
                                  max_time_between_switches=max_time_repeat,
                                  switch_cost=ConstantSwitchCost(value=jnp.array(0.0)),
                                  discounting=discount_factor,
                                  time_as_part_of_state=time_as_part_of_state,
                                  sim_dt=sim_dt,
                                  )

    wandb_config = dict(env_name=env_name,
                  backend=backend,
                  num_timesteps=num_timesteps,
                  episode_time=episode_length * g1_env.dt,
                  sim_dt=sim_dt,
                  control_dt=ctrl_dt,
                  control_frequency_hz=control_frequency_hz,
                  new_episode_steps=episode_length,
                  base_discount_factor=discount_factor,
                  seed=seed,
                  num_envs=num_envs,
                  num_eval_envs=num_eval_envs,
                  entropy_cost=entropy_cost,
                  unroll_length=unroll_length,
                  num_minibatches=num_minibatches,
                  num_updates_per_batch=num_updates_per_batch,
                  policy_hidden_layer_sizes=policy_hidden_layer_sizes,
                  critic_hidden_layer_sizes=critic_hidden_layer_sizes,
                  batch_size=batch_size,
                  reward_scaling=reward_scaling,
                  switch_cost_wrapper=switch_cost_wrapper,
                  switch_cost=switch_cost,
                  max_time_repeat=max_time_repeat,
                  time_as_part_of_state=time_as_part_of_state,
                  num_final_evals=num_final_evals,
                  min_time_repeat=min_time_repeat,
                  learning_rate=learning_rate,
                  action_repeat=action_repeat,
                  value_obs_key=value_obs_key,
                  policy_obs_key=policy_obs_key,
                  max_grad_norm=max_grad_norm,
                  num_resets_per_eval=num_resets_per_eval,
                  normalize_observations=normalize_observations,
                  clipping_epsilon=0.3,
                  gae_lambda=0.95,
                  base_dt_divisor=base_dt_divisor,
                  )
    if policy_path is not None and wandb_run_id is not None:
        # Eval-only mode: log to a separate project named <project>_Eval and
        # name the run after the original training run for traceability.
        wandb.init(
            project=project_name + '_Eval',
            name=wandb_run_id,
            config={**wandb_config, 'train_run_id': wandb_run_id},
        )
    elif switch_cost_wrapper:
        wandb.init(
            project=project_name,
            group=f"max_actions{max_time_repeat}",
            dir='/cluster/scratch/' + ENTITY,
            config=wandb_config,
        )
    else:
        wandb.init(
            project=project_name,
            dir='/cluster/scratch/' + ENTITY,
            config=wandb_config,
        )

    if switch_cost_wrapper:
        optimizer = PPO(
            environment=env,
            eval_environment=eval_env,
            num_timesteps=num_timesteps,
            episode_length=episode_length,
            action_repeat=action_repeat,
            num_envs=num_envs,
            num_eval_envs=num_eval_envs,
            lr=learning_rate,
            wd=0.,
            entropy_cost=entropy_cost,
            unroll_length=unroll_length,
            discounting=discount_factor,
            batch_size=batch_size,
            num_minibatches=num_minibatches,
            num_updates_per_batch=num_updates_per_batch,
            num_evals=num_evals,
            normalize_observations=normalize_observations,
            reward_scaling=reward_scaling,
            max_grad_norm=max_grad_norm,
            clipping_epsilon=0.3,
            gae_lambda=0.95,
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,
            policy_activation=swish,
            critic_hidden_layer_sizes=critic_hidden_layer_sizes,
            critic_activation=swish,
            deterministic_eval=True,
            normalize_advantage=True,
            wandb_logging=True,
            non_equidistant_time=True,
            min_time_between_switches=min_time_repeat,
            max_time_between_switches=max_time_repeat,
            randomization_fn=randomization_fn,
            policy_obs_key=policy_obs_key,
            value_obs_key=value_obs_key,
            seed=seed,
        )
    else:
        optimizer = PPO(
            environment=g1_env,
            num_timesteps=num_timesteps,
            episode_length=episode_length,
            action_repeat=action_repeat,
            num_envs=num_envs,
            num_eval_envs=num_eval_envs,
            lr=learning_rate,
            wd=0.,
            entropy_cost=entropy_cost,
            unroll_length=unroll_length,
            discounting=discount_factor,
            batch_size=batch_size,
            num_minibatches=num_minibatches,
            num_updates_per_batch=num_updates_per_batch,
            num_evals=num_evals,
            normalize_observations=normalize_observations,
            reward_scaling=reward_scaling,
            max_grad_norm=max_grad_norm,
            clipping_epsilon=0.3,
            gae_lambda=0.95,
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,
            policy_activation=swish,
            critic_hidden_layer_sizes=critic_hidden_layer_sizes,
            critic_activation=swish,
            deterministic_eval=True,
            normalize_advantage=True,
            wandb_logging=True,
            randomization_fn=randomization_fn,
            policy_obs_key=policy_obs_key,
            value_obs_key=value_obs_key,
            seed=seed,
        )

    xdata, ydata = [], []
    times = [datetime.now()]

    def progress(num_steps, metrics):
        times.append(datetime.now())
        xdata.append(num_steps)
        ydata.append(metrics['eval/episode_reward'])
        plt.xlabel('# environment steps')
        plt.ylabel('reward per episode')
        plt.plot(xdata, ydata)
        plt.show()

    if policy_path is not None:
        print(f'Loading policy from {policy_path}')
        with open(policy_path, 'rb') as f:
            policy_params = cloudpickle.load(f)
        print('Policy loaded.')
    else:
        print('Before inference')
        policy_params, metrics = optimizer.run_training(key=jr.PRNGKey(seed), progress_fn=progress)
        print('After inference')
        save_policy(policy_params)
        print("Policy saved to wandb!")

    ########################## Policy Rollout ##########################
    ####################################################################

    print(f'Starting with rollout')
    if switch_cost_wrapper:
        env_cfg = registry.get_default_config(env_name)
        if isinstance(env_cfg, dict):
            env_cfg['impl'] = 'jax'
        elif hasattr(env_cfg, 'impl'):
            env_cfg.impl = 'jax'
        if hasattr(env_cfg, 'sim_config'):
            if isinstance(env_cfg.sim_config, dict):
                env_cfg.sim_config['impl'] = 'jax'
            elif hasattr(env_cfg.sim_config, 'impl'):
                env_cfg.sim_config.impl = 'jax'
        if hasattr(env_cfg, 'pert_config'):
            env_cfg.pert_config.enable = perturb
        _apply_reward_overrides(env_cfg)
        eval_env = _load_g1_env(env_name, env_cfg)
        eval_env = IHSwitchCostWrapper(env=eval_env,
                                  episode_steps=episode_length,
                                  min_time_between_switches=min_time_repeat,
                                  max_time_between_switches=max_time_repeat,
                                  switch_cost=ConstantSwitchCost(value=jnp.array(0.0)),
                                  discounting=1.0,
                                  time_as_part_of_state=time_as_part_of_state,
                                  sim_dt=env_cfg.sim_dt)
        jit_reset = jax.jit(eval_env.reset)
        jit_step = jax.jit(eval_env.step)
        jit_inference_fn = jax.jit(optimizer.make_policy(policy_params, deterministic=True))

        from mujoco_playground._src.gait import draw_joystick_command

        # Command amplitude used purely to scale the on-screen joystick arrow.
        cmd_amp = jnp.array(env_cfg.command_config.a)

        seeds = range(10)
        for i in seeds:
            rng = jax.random.PRNGKey(i)
            rollout = []
            modify_scene_fns = []

            swing_peak = []
            rewards = []
            linvel = []
            angvel = []
            track = []
            foot_vel = []
            rews = []
            contact = []
            num_steps = 0
            time_predictions = []

            state = jit_reset(rng)
            # Use the command sampled by the env (matches training distribution).
            command = state.info["command"]
            env_steps = 0
            total_reward = 0.0
            disc_switch_sum = 0.0
            while env_steps < env_cfg.episode_length:
                act_rng, rng = jax.random.split(rng)
                ctrl, _ = jit_inference_fn(state.obs, act_rng)
                state = jit_step(state, ctrl)
                num_steps += 1
                predicted_time = eval_env.compute_steps(pseudo_time=ctrl[-1])
                time_predictions.append(predicted_time)
                # env_steps is T_k (elapsed inner steps when this switch fires)
                disc_switch_sum += float(discount_factor ** env_steps)
                env_steps += predicted_time
                rews.append(
                    {k: v for k, v in state.metrics.items() if k.startswith("reward/")}
                )
                total_reward += state.reward
                rollout.append(state)

                if "swing_peak" in state.info:
                    swing_peak.append(state.info["swing_peak"])
                rewards.append(
                    {k[7:]: v for k, v in state.metrics.items() if k.startswith("reward/")}
                )
                linvel.append(eval_env.env.get_global_linvel(state.data, G1_SENSOR_FRAME))
                angvel.append(eval_env.env.get_gyro(state.data, G1_SENSOR_FRAME))
                track.append(
                    eval_env.env._reward_tracking_lin_vel(
                        state.info["command"],
                        eval_env.env.get_local_linvel(state.data, G1_SENSOR_FRAME),
                    )
                )

                feet_vel = state.data.sensordata[eval_env.env._foot_linvel_sensor_adr]
                vel_xy = feet_vel[..., :2]
                vel_norm = jnp.sqrt(jnp.linalg.norm(vel_xy, axis=-1))
                foot_vel.append(vel_norm)

                if "last_contact" in state.info:
                    contact.append(state.info["last_contact"])

                xyz = np.array(state.data.xpos[eval_env.env._torso_body_id])
                xyz += np.array([0, 0, 0.3])
                x_axis = state.data.xmat[eval_env.env._torso_body_id, 0]
                yaw = -np.arctan2(x_axis[1], x_axis[0])
                modify_scene_fns.append(
                    functools.partial(
                        draw_joystick_command,
                        cmd=state.info["command"],
                        xyz=xyz,
                        theta=yaw,
                        scl=abs(state.info["command"][0]) / cmd_amp[0],
                    )
                )

            action_steps = list(range(len(time_predictions)))
            plt.figure(figsize=(10, 6))
            plt.plot(action_steps, time_predictions, marker='o', linestyle='-', color='b')
            plt.xlabel('Control step')
            plt.ylabel('Hold duration (inner steps)')
            plt.title('Hold predictions')
            disc_switch_ratio = disc_switch_sum * (1 - discount_factor)
            log_dict = {
                f'Results_{i}_random/Total reward':       total_reward,
                f'Results_{i}_random/Number of actions':  num_steps,
                f'Results_{i}_random/disc_switch_ratio':  disc_switch_ratio,
                f'Results_{i}_random/Command x_vel':      float(command[0]),
                f'Results_{i}_random/Command y_vel':      float(command[1]),
                f'Results_{i}_random/Command yaw_vel':    float(command[2]),
                f'Results_{i}_random/Time Prediction Plot': wandb.Image(plt),
            }

            # Render a video for seed 0 using simulation_step so we capture all
            # inner env states at a constant ctrl_dt frame rate.
            if i == 0:
                print('Rendering video for seed 0 (using simulation_step)...')
                try:
                    video_modify_fns = []
                    inner_states = rollout_video_tarc(
                        eval_env, jit_reset, jit_inference_fn,
                        env_cfg.episode_length, jax.random.PRNGKey(0),
                        video_modify_fns,
                    )
                    if inner_states:
                        frames = eval_env.env.render(
                            inner_states, camera='track', height=480, width=640,
                            modify_scene_fns=video_modify_fns,
                        )
                        vid_path = f'/tmp/g1_tarc_seed{i}.mp4'
                        save_video_mp4(frames, vid_path, fps=VIDEO_FPS)
                        log_dict[f'Results_{i}_random/Video'] = wandb.Video(vid_path, fps=VIDEO_FPS, format='mp4')
                        print(f'Video saved to {vid_path} and logged to wandb.')
                except Exception as e:
                    print(f'Video generation failed (skipping): {e}')
                plt.close('all')

            wandb.log(log_dict)
            print(f"Command: x_vel={float(command[0]):.2f}, y_vel={float(command[1]):.2f}, yaw_vel={float(command[2]):.2f}")
            print(f"The agent took {num_steps} actions | disc_switch_ratio={disc_switch_ratio:.4f}")
            print(f"Agent got {total_reward} reward")
    else:
        env_cfg = registry.get_default_config(env_name)
        if isinstance(env_cfg, dict):
            env_cfg['impl'] = 'jax'
        elif hasattr(env_cfg, 'impl'):
            env_cfg.impl = 'jax'
        if hasattr(env_cfg, 'sim_config'):
            if isinstance(env_cfg.sim_config, dict):
                env_cfg.sim_config['impl'] = 'jax'
            elif hasattr(env_cfg.sim_config, 'impl'):
                env_cfg.sim_config.impl = 'jax'
        if hasattr(env_cfg, 'pert_config'):
            env_cfg.pert_config.enable = perturb
        _apply_reward_overrides(env_cfg)
        if base_dt_divisor > 1:
            env_cfg.ctrl_dt *= base_dt_divisor
            print(f"Lowering eval control frequency: new ctrl_dt = {env_cfg.ctrl_dt} ({1.0 / env_cfg.ctrl_dt:.2f} Hz)")
            env_cfg.episode_length = env_cfg.episode_length // base_dt_divisor
            print(f"Adjusted episode_length = {env_cfg.episode_length} (same physical time: {env_cfg.episode_length * env_cfg.ctrl_dt:.1f}s)")
        eval_env = _load_g1_env(env_name, env_cfg)
        env = g1_env
        jit_reset = jax.jit(eval_env.reset)
        jit_step = jax.jit(eval_env.step)
        jit_inference_fn = jax.jit(optimizer.make_policy(policy_params, deterministic=True))

        from mujoco_playground._src.gait import draw_joystick_command

        cmd_amp = jnp.array(env_cfg.command_config.a)

        seeds = range(10)
        for i in seeds:
            rng = jax.random.PRNGKey(i)
            rollout = []
            modify_scene_fns = []

            swing_peak = []
            rewards = []
            linvel = []
            angvel = []
            track = []
            foot_vel = []
            rews = []
            contact = []

            state = jit_reset(rng)
            command = state.info["command"]
            num_steps = 0
            total_reward = 0
            while num_steps < env_cfg.episode_length:
                act_rng, rng = jax.random.split(rng)
                ctrl, _ = jit_inference_fn(state.obs, act_rng)
                state = jit_step(state, ctrl)
                num_steps += 1
                rews.append(
                    {k: v for k, v in state.metrics.items() if k.startswith("reward/")}
                )
                rollout.append(state)
                if "swing_peak" in state.info:
                    swing_peak.append(state.info["swing_peak"])
                rewards.append(
                    {k[7:]: v for k, v in state.metrics.items() if k.startswith("reward/")}
                )
                total_reward += state.reward
                linvel.append(env.get_global_linvel(state.data, G1_SENSOR_FRAME))
                angvel.append(env.get_gyro(state.data, G1_SENSOR_FRAME))
                track.append(
                    env._reward_tracking_lin_vel(
                        state.info["command"],
                        env.get_local_linvel(state.data, G1_SENSOR_FRAME),
                    )
                )

                feet_vel = state.data.sensordata[env._foot_linvel_sensor_adr]
                vel_xy = feet_vel[..., :2]
                vel_norm = jnp.sqrt(jnp.linalg.norm(vel_xy, axis=-1))
                foot_vel.append(vel_norm)

                if "last_contact" in state.info:
                    contact.append(state.info["last_contact"])

                xyz = np.array(state.data.xpos[env._torso_body_id])
                xyz += np.array([0, 0, 0.3])
                x_axis = state.data.xmat[env._torso_body_id, 0]
                yaw = -np.arctan2(x_axis[1], x_axis[0])
                modify_scene_fns.append(
                    functools.partial(
                        draw_joystick_command,
                        cmd=state.info["command"],
                        xyz=xyz,
                        theta=yaw,
                        scl=abs(state.info["command"][0]) / cmd_amp[0],
                    )
                )

            avg_control_freq = num_steps / (episode_length * ctrl_dt)
            log_dict = {
                f'Results_{i}_random/Total reward':            total_reward,
                f'Results_{i}_random/Number of actions':       num_steps,
                f'Results_{i}_random/Avg control frequency (Hz)': avg_control_freq,
                f'Results_{i}_random/Command x_vel':           float(command[0]),
                f'Results_{i}_random/Command y_vel':           float(command[1]),
                f'Results_{i}_random/Command yaw_vel':         float(command[2]),
            }

            # Render video for seed 0: rollout states are already at ctrl_dt.
            if i == 0 and rollout:
                print('Rendering video for seed 0 (baseline)...')
                try:
                    frames = eval_env.render(
                        rollout, camera='track', height=480, width=640,
                        modify_scene_fns=modify_scene_fns,
                    )
                    vid_path = f'/tmp/g1_baseline_seed{i}.mp4'
                    save_video_mp4(frames, vid_path, fps=VIDEO_FPS)
                    log_dict[f'Results_{i}_random/Video'] = wandb.Video(vid_path, fps=VIDEO_FPS, format='mp4')
                    print(f'Video saved to {vid_path} and logged to wandb.')
                except Exception as e:
                    print(f'Video generation failed (skipping): {e}')
                plt.close('all')

            wandb.log(log_dict)
            print(f"Command: x_vel={float(command[0]):.2f}, y_vel={float(command[1]):.2f}, yaw_vel={float(command[2]):.2f}")
            print(f"Agent got {total_reward} reward at avg {avg_control_freq:.1f} Hz ({num_steps} actions)")

    wandb.finish()


def main(args):
    experiment(env_name=args.env_name,
               backend=args.backend,
               project_name=args.project_name,
               seed=args.seed,
               num_eval_envs=args.num_eval_envs,
               switch_cost_wrapper=bool(args.switch_cost_wrapper),
               switch_cost=args.switch_cost,
               max_time_repeat=args.max_time_repeat,
               time_as_part_of_state=bool(args.time_as_part_of_state),
               num_final_evals=args.num_final_evals,
               min_time_repeat=args.min_time_repeat,
               perturb=bool(args.perturb),
               base_dt_divisor=args.base_dt_divisor,
               policy_path=args.policy_path,
               wandb_run_id=args.wandb_run_id,
               )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='G1JoystickFlatTerrain')
    parser.add_argument('--backend', type=str, default='generalized')
    parser.add_argument('--project_name', type=str, default='G1_TARC')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_eval_envs', type=int, default=128)
    parser.add_argument('--switch_cost_wrapper', type=int, default=1)
    parser.add_argument('--switch_cost', type=float, default=0.005)
    parser.add_argument('--max_time_repeat', type=int, default=4)
    parser.add_argument('--min_time_repeat', type=int, default=1)
    parser.add_argument('--time_as_part_of_state', type=int, default=1)
    parser.add_argument('--num_final_evals', type=int, default=1)
    parser.add_argument('--perturb', type=int, default=0)
    parser.add_argument('--base_dt_divisor', type=int, default=1)
    parser.add_argument('--policy_path', type=str, default=None,
                        help='Path to a saved policy pkl file. If given, skips training.')
    parser.add_argument('--wandb_run_id', type=str, default=None,
                        help='W&B run ID to resume when loading a policy (eval-only mode).')
    args = parser.parse_args()
    main(args)
