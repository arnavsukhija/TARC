import functools
import math
import time
from functools import partial
from typing import Any, Tuple, Sequence, Callable, Mapping, Union, Optional

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import optax
import wandb
from brax import envs, base
from brax.training import acting
from brax.training import networks
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.types import Params
from jax.lax import scan
from jaxtyping import PyTree

from mujoco_playground._src.wrapper import wrap_for_brax_training as wrap_for_training

from wtc.agents.ppo.losses_new import PPOLoss, PPONetworkParams
from wtc.agents.ppo.ppo_network import PPONetworksModel, make_inference_fn
from mbpo.optimizers.policy_optimizers.sac.utils import gradient_update_fn, metrics_to_float

Metrics = types.Metrics
Transition = types.Transition
InferenceParams = Tuple[running_statistics.NestedMeanStd, Params]

ReplayBufferState = Any


@chex.dataclass
class TrainingState:
    """Contains training state for the learner."""
    optimizer_state: optax.OptState
    params: PPONetworkParams
    normalizer_params: running_statistics.RunningStatisticsState
    env_steps: jnp.ndarray

    def get_policy_params(self):
        return self.normalizer_params, self.params.policy


def _maybe_wrap_env(
        env: envs.Env,
        wrap_env: bool,
        num_envs: int,
        episode_length: Optional[int],
        action_repeat: int,
        local_device_count: int,
        key_env: jr.PRNGKey,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
):
    """Wraps the environment for training/eval if wrap_env is True."""
    if not wrap_env:
        return env
    if episode_length is None:
        raise ValueError('episode_length must be specified')
    v_randomization_fn = None
    if randomization_fn is not None:
        randomization_batch_size = num_envs // local_device_count
        # all devices gets the same randomization rng
        randomization_rng = jax.random.split(key_env, randomization_batch_size)
        v_randomization_fn = functools.partial(
            randomization_fn, rng=randomization_rng
        )
    env = wrap_for_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )  # pytype: disable=wrong-keyword-args
    return env


class PPO:
    def __init__(self,
                 environment: envs.Env,
                 num_timesteps: int,
                 episode_length: int,
                 eval_environment: envs.Env = None,
                 action_repeat: int = 1,
                 num_envs: int = 1,
                 num_eval_envs: int = 128,
                 lr: float = 1e-4,
                 wd: float = 1e-5,
                 max_grad_norm: float = 1e5,
                 entropy_cost: float = 1e-4,
                 discounting: float = 0.9,
                 seed: int = 0,
                 unroll_length: int = 10,
                 batch_size: int = 32,
                 num_minibatches: int = 16,
                 num_updates_per_batch: int = 2,
                 num_evals: int = 1,
                 normalize_observations: bool = False,
                 non_equidistant_time: bool = False,
                 reward_scaling: float = 1.,
                 clipping_epsilon: float = .3,
                 gae_lambda: float = .95,
                 deterministic_eval: bool = False,
                 normalize_advantage: bool = True,
                 policy_hidden_layer_sizes: Sequence[int] = (64, 64, 64),
                 policy_activation: networks.ActivationFn = nn.swish,
                 critic_hidden_layer_sizes: Sequence[int] = (64, 64, 64),
                 critic_activation: networks.ActivationFn = nn.swish,
                 wandb_logging: bool = False,
                 return_best_model: bool = False,
                 continuous_discounting: float = 0,
                 min_time_between_switches: float = 0,
                 max_time_between_switches: float = 0,
                 policy_obs_key: str = 'state',
                 value_obs_key: str = 'state',
                 randomization_fn: Optional[
                     Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
                 ] = None
                 ):
        self.wandb_logging = wandb_logging
        self.return_best_model = return_best_model
        self.episode_length = episode_length
        self.action_repeat = action_repeat
        self.num_timesteps = num_timesteps
        self.deterministic_eval = deterministic_eval
        self.normalize_advantage = normalize_advantage
        self.gae_lambda = gae_lambda
        self.clipping_epsilon = clipping_epsilon
        self.reward_scaling = reward_scaling
        self.normalize_observations = normalize_observations
        self.num_evals = num_evals
        self.num_updates_per_batch = num_updates_per_batch
        self.num_minibatches = num_minibatches
        self.batch_size = batch_size
        self.unroll_length = unroll_length
        self.discounting = discounting
        self.entropy_cost = entropy_cost
        self.num_eval_envs = num_eval_envs
        self.num_envs = num_envs
        self.non_equidistant_time = non_equidistant_time
        self.randomization_fn = randomization_fn
        # Set this to None unless you want to use parallelism across multiple devices.
        self._PMAP_AXIS_NAME = None

        assert batch_size * num_minibatches % num_envs == 0
        self.env_step_per_training_step = batch_size * unroll_length * num_minibatches * action_repeat
        num_evals_after_init = max(num_evals - 1, 1)
        self.num_evals_after_init = num_evals_after_init
        num_training_steps_per_epoch = math.ceil(
            num_timesteps / (num_evals_after_init * self.env_step_per_training_step))
        self.num_training_steps_per_epoch = num_training_steps_per_epoch
        self.key = jr.PRNGKey(seed)
        self.key, key_dr = jr.split(self.key)
        self.env = _maybe_wrap_env(environment, wrap_env=True,
                                   num_envs=self.num_envs, episode_length=self.episode_length,
                                   action_repeat=self.action_repeat, local_device_count=jax.local_device_count(),
                                   key_env=key_dr, randomization_fn=randomization_fn)
        # Only one local device for evaluation
        key_dr, key_dreval = jr.split(key_dr)
        self.eval_env = _maybe_wrap_env(eval_environment if eval_environment is not None else environment,
                                        wrap_env=True, num_envs=self.num_eval_envs, episode_length=self.episode_length,
                                        action_repeat=self.action_repeat,
                                        local_device_count=1, key_env=key_dreval, randomization_fn=randomization_fn)
        self.x_dim = self.env.observation_size
        self.u_dim = self.env.action_size

        def normalize_fn(batch: PyTree, _: PyTree) -> PyTree:
            return batch

        if normalize_observations:
            normalize_fn = running_statistics.normalize
        self.normalize_fn = normalize_fn

        self.ppo_networks_model = PPONetworksModel(
            x_dim=self.x_dim, u_dim=self.u_dim,
            preprocess_observations_fn=normalize_fn,
            policy_hidden_layer_sizes=policy_hidden_layer_sizes,
            policy_activation=policy_activation,
            value_hidden_layer_sizes=critic_hidden_layer_sizes,
            value_activation=critic_activation,
            policy_obs_key=policy_obs_key,
            value_obs_key=value_obs_key, )

        self.make_policy = make_inference_fn(self.ppo_networks_model.get_ppo_networks())

        self.optimizer = optax.chain(
            optax.clip_by_global_norm(max_norm=max_grad_norm),
            optax.adamw(learning_rate=lr, weight_decay=wd)
        )

        self.ppo_loss = PPOLoss(ppo_network=self.ppo_networks_model.get_ppo_networks(),
                                entropy_cost=self.entropy_cost,
                                discounting=self.discounting,
                                reward_scaling=self.reward_scaling,
                                gae_lambda=self.gae_lambda,
                                clipping_epsilon=self.clipping_epsilon,
                                normalize_advantage=self.normalize_advantage,
                                continuous_discounting=continuous_discounting,
                                min_time_between_switches=min_time_between_switches,
                                max_time_between_switches=max_time_between_switches,
                                non_equidistant_time=self.non_equidistant_time,
                                )

        self.ppo_update = gradient_update_fn(self.ppo_loss.loss, self.optimizer, pmap_axis_name=self._PMAP_AXIS_NAME,
                                             has_aux=True)

    def minibatch_step(self,
                       carry,
                       data: types.Transition,
                       normalizer_params: running_statistics.RunningStatisticsState):
        # Performs one ppo update
        optimizer_state, params, key = carry
        key, key_loss = jr.split(key)
        (_, metrics), params, optimizer_state = self.ppo_update(
            params,
            normalizer_params,
            data,
            key_loss,
            optimizer_state=optimizer_state)

        return (optimizer_state, params, key), metrics

    def sgd_step(self,
                 carry,
                 unused_t,
                 data: types.Transition,
                 normalizer_params: running_statistics.RunningStatisticsState):
        optimizer_state, params, key = carry
        key, key_perm, key_grad = jr.split(key, 3)

        def convert_data(x: chex.Array):
            x = jr.permutation(key_perm, x)
            x = jnp.reshape(x, (self.num_minibatches, -1) + x.shape[1:])
            return x

        shuffled_data = jtu.tree_map(convert_data, data)
        (optimizer_state, params, _), metrics = scan(
            partial(self.minibatch_step, normalizer_params=normalizer_params),
            (optimizer_state, params, key_grad),
            shuffled_data,
            length=self.num_minibatches)
        return (optimizer_state, params, key), metrics

    def training_step(
            self,
            carry: Tuple[TrainingState, envs.State, chex.PRNGKey],
            unused_t) -> Tuple[Tuple[TrainingState, envs.State, chex.PRNGKey], Metrics]:
        """
        1.  Performs a rollout of length `self.unroll_length` using the current policy.
            Since there is self.num_envs environments, this will result in a self.num_envs * self.unroll_length
            collected transitions.
        """
        training_state, state, key = carry
        key_sgd, key_generate_unroll, new_key = jr.split(key, 3)

        # Deterministic is set by default to False
        policy = self.make_policy((training_state.normalizer_params, training_state.params.policy))

        def f(carry, unused_t):
            current_state, current_key = carry
            current_key, next_key = jr.split(current_key)
            next_state, data = acting.generate_unroll(
                self.env,
                current_state,
                policy,
                current_key,
                self.unroll_length,
                extra_fields=('truncation',))
            return (next_state, next_key), data

        (state, _), data = scan(
            f, (state, key_generate_unroll), (),
            length=self.batch_size * self.num_minibatches // self.num_envs)
        # Have leading dimensions (batch_size * num_minibatches, unroll_length)
        data = jtu.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jtu.tree_map(lambda x: jnp.reshape(x, (-1,) + x.shape[2:]),
                            data)
        assert data.discount.shape[1:] == (self.unroll_length,)

        # Update normalization params and normalize observations.
        normalizer_params = running_statistics.update(
            training_state.normalizer_params,
            data.observation,
            pmap_axis_name=self._PMAP_AXIS_NAME)

        # Perform self.num_updates_per_batch calls of self.sgd_step
        (optimizer_state, params, _), metrics = scan(
            partial(
                self.sgd_step, data=data, normalizer_params=normalizer_params),
            (training_state.optimizer_state, training_state.params, key_sgd), (),
            length=self.num_updates_per_batch)

        new_training_state = TrainingState(
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            env_steps=training_state.env_steps + self.env_step_per_training_step)
        return (new_training_state, state, new_key), metrics

    def training_epoch(
            self,
            training_state: TrainingState,
            state: envs.State,
            key: chex.PRNGKey) -> Tuple[TrainingState, envs.State, Metrics]:
        """
        Performs self.num_training_steps_per_epoch calls of self.training_step functions.
        """
        (training_state, state, _), loss_metrics = scan(
            self.training_step, (training_state, state, key), (),
            length=self.num_training_steps_per_epoch)
        loss_metrics = jtu.tree_map(jnp.mean, loss_metrics)
        return training_state, state, loss_metrics

    def training_epoch_with_timing(
            self,
            training_state: TrainingState,
            env_state: envs.State,
            key: chex.PRNGKey) -> Tuple[TrainingState, envs.State, Metrics]:
        t = time.time()
        training_state, env_state, metrics = self.training_epoch(training_state, env_state, key)

        metrics = jtu.tree_map(jnp.mean, metrics)
        epoch_training_time = time.time() - t
        sps = (self.num_training_steps_per_epoch * self.env_step_per_training_step) / epoch_training_time
        metrics = {
            'training/sps': jnp.array(sps),
            **{f'training/{name}': jnp.array(value) for name, value in metrics.items()}
        }
        return training_state, env_state, metrics

    def init_training_state(self, key: chex.PRNGKey) -> TrainingState:
        keys = jr.split(key)
        init_params = PPONetworkParams(
            policy=self.ppo_networks_model.get_policy_network().init(keys[0]),
            value=self.ppo_networks_model.get_value_network().init(keys[1]))
        # get tbe observatiion spec for the state and privileged state
        obs_spec = {
            k: specs.Array(shape, jnp.float32)
            for k, shape in self.x_dim.items()
            if not k.startswith("pixels/")
        }
        training_state = TrainingState(
            optimizer_state=self.optimizer.init(init_params),
            params=init_params,
            normalizer_params=running_statistics.init_state(obs_spec),
            env_steps=0,
        )
        return training_state

    def run_training(self, key: chex.PRNGKey, progress_fn: Callable[[int, Metrics], None] = lambda *args: None):
        key, subkey = jr.split(key)
        training_state = self.init_training_state(subkey)
        key, rb_key, env_key, eval_key = jr.split(key, 4)

        # Initialize initial env state
        env_keys = jr.split(env_key, self.num_envs)
        env_state = self.env.reset(env_keys)

        # we use a separate evaluation env here
        evaluator = acting.Evaluator(
            self.eval_env, partial(self.make_policy, deterministic=self.deterministic_eval),
            num_eval_envs=self.num_eval_envs, episode_length=self.episode_length, action_repeat=self.action_repeat,
            key=eval_key)

        # Run initial eval
        all_metrics = []
        metrics = {}
        highest_eval_episode_reward = jnp.array(-jnp.inf)
        best_params = (training_state.normalizer_params, training_state.params.policy)
        if self.num_evals > 1:
            metrics = evaluator.run_evaluation((training_state.normalizer_params, training_state.params.policy),
                                               training_metrics={})

            if self.wandb_logging:
                metrics = metrics_to_float(metrics)
                wandb.log(metrics)
            all_metrics.append(metrics)
            progress_fn(0, metrics)

        # Create and initialize the replay buffer.
        key, prefill_key = jr.split(key)

        current_step = 0
        for _ in range(self.num_evals_after_init):
            if self.wandb_logging:
                wandb.log(metrics_to_float({'step': current_step}))

            # Optimization
            key, epoch_key = jr.split(key)
            training_state, env_state, training_metrics = self.training_epoch_with_timing(training_state, env_state,
                                                                                          epoch_key)
            current_step = training_state.env_steps

            # Eval and logging
            # Run evals.
            metrics = evaluator.run_evaluation((training_state.normalizer_params, training_state.params.policy),
                                               training_metrics)
            if metrics['eval/episode_reward'] > highest_eval_episode_reward:
                highest_eval_episode_reward = metrics['eval/episode_reward']
                best_params = (training_state.normalizer_params, training_state.params.policy)
            if self.wandb_logging:
                metrics = metrics_to_float(metrics)
                wandb.log(metrics)
            all_metrics.append(metrics)
            progress_fn(current_step, metrics)

        total_steps = current_step
        # assert total_steps >= self.num_timesteps
        last_params = (training_state.normalizer_params, training_state.params.policy)

        if self.return_best_model:
            params_to_return = best_params
        else:
            params_to_return = last_params

        # If there were no mistakes the training_state should still be identical on all
        # devices.
        if self.wandb_logging:
            wandb.log(metrics_to_float({'total steps': total_steps}))
        return params_to_return, all_metrics
