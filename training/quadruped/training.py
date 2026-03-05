import argparse
import datetime
import functools
import os
import cloudpickle
from datetime import datetime

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


from jax import config

config.update("jax_debug_nans", True)

ENTITY = 'asukhija'

# fix rendering
os.environ["MUJOCO_GL"] = "egl"


def save_policy(policy_params):
    if wandb.run is None:
        raise RuntimeError("wandb.run is not initialized. Ensure wandb.init() is called before logging artifacts.")

    # Ensure the 'Policies' directory inside the wandb run directory exists
    directory = os.path.join(os.getcwd(), 'Policies')
    if not os.path.exists(directory):
        os.makedirs(directory)

    policy_path = os.path.join(directory, f"policy_params_{wandb.run.id}.pkl")

    try:

        with open(policy_path, "wb") as f:
            cloudpickle.dump(policy_params, f)

        try:
            with open(policy_path, "rb") as f:
                loaded_policy = cloudpickle.load(f)
            print("Successfully loaded policy from file for verification.")
        except Exception as e:
            print(f"Error loading policy file for verification: {e}")
            return  # Stop the upload if the file is invalid.

        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"File not found: {policy_path}")

        wandb.save(policy_path, wandb.run.dir)

        print(f"Successfully saved and uploaded {policy_path} to Weights & Biases.")

    except Exception as e:
        print(f"An error occurred during policy upload: {e}")
    print("Policy saved to wandb!")
def experiment(env_name: str = 'Go1JoystickFlatTerrain',
               backend: str = 'generalized',
               project_name: str = 'GPUSpeedTest',
               seed: int = 0,
               num_eval_envs: int = 128,
               switch_cost_wrapper: bool = False,
               switch_cost: float = 0.1,
               max_time_repeat: int = 10,
               min_time_repeat: int = 1,
               time_as_part_of_state: bool = True,
               num_final_evals: int = 10,
               perturb: bool = False,
               ):

    env_cfg = registry.get_default_config(env_name)
    if perturb:
        env_cfg.pert_config.enable = True
    go1_env = registry.load(env_name, env_cfg)
    ppo_params = locomotion_params.brax_ppo_config(env_name)
    ppo_config = dict(ppo_params)
    action_repeat = ppo_config['action_repeat']
    batch_size = ppo_config['batch_size']
    discount_factor = ppo_config['discounting']
    entropy_cost = ppo_config['entropy_cost']
    episode_length = ppo_config['episode_length']
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
    num_timesteps = ppo_config['num_timesteps']*3 ## train longer for 600mil time steps
    num_updates_per_batch = ppo_config['num_updates_per_batch']
    reward_scaling = ppo_config['reward_scaling']
    unroll_length = ppo_config['unroll_length']
    sim_dt = env_cfg['sim_dt']
    ctrl_dt = env_cfg['ctrl_dt']


    randomization_fn = registry.get_domain_randomizer(env_name)

    if switch_cost_wrapper:
        env = IHSwitchCostWrapper(env=go1_env,
                                  episode_steps=episode_length,
                                  min_time_between_switches=min_time_repeat,
                                  max_time_between_switches=max_time_repeat,
                                  switch_cost=ConstantSwitchCost(value=jnp.array(switch_cost)),
                                  discounting=discount_factor,
                                  time_as_part_of_state=time_as_part_of_state,
                                  sim_dt = sim_dt,
                                  )
        eval_env = IHSwitchCostWrapper(env=registry.load(env_name, env_cfg),
                                  episode_steps=episode_length,
                                  min_time_between_switches=min_time_repeat,
                                  max_time_between_switches=max_time_repeat,
                                  switch_cost=ConstantSwitchCost(value=jnp.array(0.0)),
                                  discounting=1.0,
                                  time_as_part_of_state=time_as_part_of_state,
                                  sim_dt = sim_dt,
                                  )

    config = dict(env_name=env_name,
                  backend=backend,
                  num_timesteps=num_timesteps,
                  episode_time=episode_length * go1_env.dt,
                  sim_dt=sim_dt,
                  control_dt=ctrl_dt,
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
                  action_repeat = action_repeat,
                  value_obs_key = value_obs_key,
                  policy_obs_key = policy_obs_key,
                  max_grad_norm = max_grad_norm,
                  num_resets_per_eval = num_resets_per_eval,
                  normalize_observations = normalize_observations,
                  clipping_epsilon = 0.3,
                  gae_lambda = 0.95,
                  )
    if switch_cost_wrapper:
        wandb.init(
            project=project_name,
            group=f"max_actions{max_time_repeat}",
            dir='/cluster/scratch/' + ENTITY,
            config=config,
        )
    else:
        wandb.init(
            project=project_name,
            dir='/cluster/scratch/' + ENTITY,
            config=config,
        )
    times = [datetime.now()]

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
            non_equidistant_time = True,
            min_time_between_switches=min_time_repeat,
            max_time_between_switches=max_time_repeat,
            randomization_fn=randomization_fn,
            policy_obs_key=policy_obs_key,
            value_obs_key=value_obs_key,
            seed = seed,
        )
    else:
        optimizer = PPO(
            environment=go1_env,
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
            seed = seed,
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

    print('Before inference')
    policy_params, metrics = optimizer.run_training(key=jr.PRNGKey(seed), progress_fn=progress)
    print('After inference')
    save_policy(policy_params)
    print("Policy saved to wandb!")

    ########################## Policy Rollout ##########################
    ################################################################

    print(f'Starting with rollout')
    if switch_cost_wrapper:
        env_cfg = registry.get_default_config(env_name)
        env_cfg.pert_config.enable = perturb
        env_cfg.command_config.a = [1.5, 0.8, 2 * jnp.pi]
        eval_env = registry.load(env_name, config=env_cfg)
        eval_env = IHSwitchCostWrapper(env=eval_env,
                                  episode_steps=episode_length,
                                  min_time_between_switches=min_time_repeat,
                                  max_time_between_switches=max_time_repeat,
                                  switch_cost=ConstantSwitchCost(value=jnp.array(0.0)),
                                  discounting=1.0,
                                  time_as_part_of_state=time_as_part_of_state,
                                  sim_dt = env_cfg.sim_dt)
        jit_reset = jax.jit(eval_env.reset)
        jit_step = jax.jit(eval_env.step)
        jit_inference_fn = jax.jit(optimizer.make_policy(policy_params, deterministic=True))

        from mujoco_playground._src.gait import draw_joystick_command

        x_vel = 0.0  # @param {type: "number"}
        y_vel = 0.0  # @param {type: "number"}
        yaw_vel = 3.14  # @param {type: "number"}

        seeds = [42,43,44]
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
            command = jnp.array([x_vel, y_vel, yaw_vel])
            num_steps = 0
            time_predictions = []

            state = jit_reset(rng)
            state.info["command"] = command
            env_steps = 0
            total_reward = 0.0
            while env_steps < env_cfg.episode_length:
                act_rng, rng = jax.random.split(rng)
                ctrl, _ = jit_inference_fn(state.obs, act_rng)
                time_predictions.append(ctrl[-1])
                state= jit_step(state, ctrl)
                num_steps += 1
                predicted_time = env.compute_steps(pseudo_time=ctrl[-1])
                time_predictions.append(predicted_time)
                env_steps += predicted_time
                state.info["command"] = command
                rews.append(
                    {k: v for k, v in state.metrics.items() if k.startswith("reward/")}
                )
                total_reward += state.reward
                rollout.append(state)
                swing_peak.append(state.info["swing_peak"])
                rewards.append(
                    {k[7:]: v for k, v in state.metrics.items() if k.startswith("reward/")}
                )
                linvel.append(env.env.get_global_linvel(state.data))
                angvel.append(env.env.get_gyro(state.data))
                track.append(
                    env.env._reward_tracking_lin_vel(
                        state.info["command"], env.env.get_local_linvel(state.data)
                    )
                )

                feet_vel = state.data.sensordata[env.env._foot_linvel_sensor_adr]
                vel_xy = feet_vel[..., :2]
                vel_norm = jnp.sqrt(jnp.linalg.norm(vel_xy, axis=-1))
                foot_vel.append(vel_norm)

                contact.append(state.info["last_contact"])

                xyz = np.array(state.data.xpos[env.env._torso_body_id])
                xyz += np.array([0, 0, 0.2])
                x_axis = state.data.xmat[env.env._torso_body_id, 0]
                yaw = -np.arctan2(x_axis[1], x_axis[0])
                modify_scene_fns.append(
                    functools.partial(
                        draw_joystick_command,
                        cmd=state.info["command"],
                        xyz=xyz,
                        theta=yaw,
                        scl=abs(state.info["command"][0])
                            / env_cfg.command_config.a[0],
                    )
                )
            action_steps = list(range(len(time_predictions)))
            plt.figure(figsize=(10, 6))
            plt.plot(action_steps, time_predictions, marker='o', linestyle='-', color='b')
            plt.xlabel('Control step')
            plt.ylabel('Time Prediction')
            plt.title('Hold predictions')
            wandb.log({f'Results_{i}/Total reward': total_reward})
            wandb.log({f'Results_{i}/Number of actions': num_steps})
            wandb.log({f"Results_{i}/Time Prediction Plot": wandb.Image(plt)})
            print(f"The agent took {num_steps} actions")
            print(f"Agent got {total_reward} reward")
    else:
        # Enable perturbation in the eval env.
        env_cfg = registry.get_default_config(env_name)
        env_cfg.pert_config.enable = perturb
        env_cfg.command_config.a = [1.5, 0.8, 2 * jnp.pi]
        eval_env = registry.load(env_name, config=env_cfg)
        env = go1_env
        jit_reset = jax.jit(eval_env.reset)
        jit_step = jax.jit(eval_env.step)
        jit_inference_fn = jax.jit(optimizer.make_policy(policy_params, deterministic=True))
        from mujoco_playground._src.gait import draw_joystick_command

        x_vel = 0.0  # @param {type: "number"}
        y_vel = 0.0  # @param {type: "number"}
        yaw_vel = 3.14  # @param {type: "number"}

        seeds = [42,43,44]
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
            command = jnp.array([x_vel, y_vel, yaw_vel])

            state = jit_reset(rng)
            state.info["command"] = command
            num_steps = 0
            total_reward = 0
            while num_steps < env_cfg.episode_length:
                act_rng, rng = jax.random.split(rng)
                ctrl, _ = jit_inference_fn(state.obs, act_rng)
                state = jit_step(state, ctrl)
                num_steps += 1
                state.info["command"] = command
                rews.append(
                    {k: v for k, v in state.metrics.items() if k.startswith("reward/")}
                )
                rollout.append(state)
                swing_peak.append(state.info["swing_peak"])
                rewards.append(
                    {k[7:]: v for k, v in state.metrics.items() if k.startswith("reward/")}
                )
                total_reward += state.reward
                linvel.append(env.get_global_linvel(state.data))
                angvel.append(env.get_gyro(state.data))
                track.append(
                    env._reward_tracking_lin_vel(
                        state.info["command"], env.get_local_linvel(state.data)
                    )
                )

                feet_vel = state.data.sensordata[env._foot_linvel_sensor_adr]
                vel_xy = feet_vel[..., :2]
                vel_norm = jnp.sqrt(jnp.linalg.norm(vel_xy, axis=-1))
                foot_vel.append(vel_norm)

                contact.append(state.info["last_contact"])

                xyz = np.array(state.data.xpos[env._torso_body_id])
                xyz += np.array([0, 0, 0.2])
                x_axis = state.data.xmat[env._torso_body_id, 0]
                yaw = -np.arctan2(x_axis[1], x_axis[0])
                modify_scene_fns.append(
                    functools.partial(
                        draw_joystick_command,
                        cmd=state.info["command"],
                        xyz=xyz,
                        theta=yaw,
                        scl=abs(state.info["command"][0])
                            / env_cfg.command_config.a[0],
                    )
                )
            wandb.log({f'Results_{i}/Total reward ': total_reward})
            wandb.log({f'Results_{i}/Number of actions': num_steps})
            print(f"Agent got {total_reward} reward")

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
               )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='Go1JoystickFlatTerrain')
    parser.add_argument('--backend', type=str, default='generalized')
    parser.add_argument('--project_name', type=str, default='GPUSpeedTest')
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--num_eval_envs', type=int, default=128)
    parser.add_argument('--switch_cost_wrapper', type=int, default=1)
    parser.add_argument('--switch_cost', type=float, default=1.0)
    parser.add_argument('--max_time_repeat', type=int, default=5)
    parser.add_argument('--min_time_repeat', type=int, default=1)
    parser.add_argument('--time_as_part_of_state', type=int, default=1)
    parser.add_argument('--num_final_evals', type=int, default=10)
    parser.add_argument('--perturb', type=int, default=0)
    args = parser.parse_args()
    main(args)
