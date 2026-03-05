import functools
from abc import abstractmethod
from functools import partial
from typing import NamedTuple, Callable, Tuple, Mapping, Optional, Any

import chex
import mujoco
from jax import jit
from jaxtyping import Float, Array
import jax
import jax.numpy as jnp
from jax.lax import while_loop
import jax.tree_util as jtu
from mujoco import mjx
from mujoco_playground._src import mjx_env

EPS = 1e-10
class AugmentedPipelineState(NamedTuple):
    pipeline_state: mjx_env.State
    time: Float[Array, 'None']


class SwitchCost:
    @abstractmethod
    def __call__(self,
                 state: Float[Array, 'observation_dim'],
                 action: Float[Array, 'action_dim']) -> Float[Array, 'None']:
        pass


class ConstantSwitchCost(SwitchCost):

    def __init__(self, value: Float[Array, 'None']):
        self.value = value

    @partial(jit, static_argnums=(0,))
    def __call__(self,
                 state: Float[Array, 'observation_dim'],
                 action: Float[Array, 'action_dim']) -> Float[Array, 'None']:
        return self.value


"""An adapted version of the SwitchCostWrapper for Mujoco Environments. Here, the state is augmented by a discrete time step.
Requires the optimizer to support time-dependent discounting based on discrete time steps, not continuous time periods. 
"""
class IHSwitchCostWrapper(mjx_env.MjxEnv):
    def __init__(self,
                 env: mjx_env,
                 episode_steps: int,
                 min_time_between_switches: float,  # corresponds to tmin
                 max_time_between_switches: float | None = None,  # corresponds to tmax
                 switch_cost: SwitchCost = ConstantSwitchCost(value=jnp.array(0.1)),
                 discounting: float = 0.99,
                 time_as_part_of_state: bool = False,
                 sim_dt: float = 1,
                 ):
        self.env = env
        self._mjx_model = self.env.mjx_model
        self._mj_model = self.env.mj_model
        self._unwrapped = self.env.unwrapped
        self._xml_path = self.env.xml_path
        self.episode_steps = episode_steps
        self.num_integrator_steps = episode_steps * env.dt / sim_dt
        self.switch_cost = switch_cost
        self.min_time_between_switches = min_time_between_switches
        assert min_time_between_switches >= 1, \
            'Min time between switches must be at least 1 '  #otherwise the integration term makes no sense at all
        self.time_horizon = self.env.dt * episode_steps  #this corresponds to the T from the paper, should be
        if max_time_between_switches is None:
            max_time_between_switches = self.time_horizon
        self.max_time_between_switches = max_time_between_switches
        self.discounting = discounting
        self.time_as_part_of_state = time_as_part_of_state  #this includes the state definition, for interaction cost time is part of the state
        self.jitted_step_fn = jit(self.env.step)

    def _add_time_to_obs(self, state: mjx_env.State, time: jax.Array) -> Mapping[str, jnp.ndarray]:
        reshaped_time = time.reshape(1)
        # we handle the case where it is a state from a Mujoco Env
        state_obs = state.obs['state']
        privileged_state = state.obs['privileged_state']
        new_state_obs = jnp.concatenate([state_obs, reshaped_time])
        new_privileged_state = jnp.hstack([
            new_state_obs,
            privileged_state[state_obs.size:]
        ])
        augmented_obs = {
            'state': new_state_obs,
            'privileged_state': new_privileged_state
        }
        return augmented_obs

    def reset(self, rng: jax.Array) -> mjx_env.State:
        """
        The augmented state is represented by concatenated vector: #also includes reward (implicit in the state)
         (state, time-to-go)
        """
        state = self.env.reset(rng)
        time = jnp.array(0)
        if self.time_as_part_of_state:
            # we check whether the state observation is a jax.Array or a mapping, and extract the obs accordingly for the concatenation
            augmented_obs = self._add_time_to_obs(state, time)
            augmented_state = state.replace(obs=augmented_obs)
        else:
            augmented_pipeline_state = AugmentedPipelineState(pipeline_state=state.pipeline_state,
                                                              time=time)
            augmented_state = state.replace(pipeline_state=augmented_pipeline_state)
        return augmented_state

    def compute_time(self,
                     pseudo_time: chex.Array,
                     t_lower: chex.Array,  # pass this as time now
                     t_upper: chex.Array,  # pass this as time now
                     ) -> chex.Array:
        time_for_action = ((t_upper - t_lower) / 2.0 * pseudo_time + (
                    t_upper + t_lower) / 2.0)  #pseudo time for action is between [-1,1], we map it to tmin, tmax
        return jnp.floor(time_for_action)

    def compute_steps(self, pseudo_time: chex.Array) -> chex.Array:
        func = functools.partial(self.compute_time, t_lower=self.min_time_between_switches,
                                 t_upper=self.max_time_between_switches)
        return func(pseudo_time)

    def _get_time_and_obs(self, state: mjx_env.State) -> Tuple[jax.Array, jax.Array, jax.Array]:
        obs, time = state.obs['state'][:-1], state.obs['state'][-1]
        obs_size = obs.size
        privileged_state = jnp.concatenate([state.obs['privileged_state'][:obs_size], state.obs['privileged_state'][
                                                                                      obs_size + 1:]])  # only applicable if value state is different than policy state
        return obs, time, privileged_state

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        u, pseudo_time_for_action = action[:-1], action[-1]
        if self.time_as_part_of_state:
            obs, time, privileged_state_obs = self._get_time_and_obs(state)
        else:
            env_pipeline_state = state.pipeline_state.pipeline_state
            time = state.pipeline_state.time

        # Calculate the action time, i.e. Map pseudo_time_for_action from [-1, 1] to
        # time [self.min_time_between_switches, self.max_time_between_switches] (corresponds to number of steps now)
        steps_to_apply = self.compute_time(pseudo_time=pseudo_time_for_action,
                                           t_lower=self.min_time_between_switches,
                                           t_upper=self.max_time_between_switches,
                                           )

        done = steps_to_apply >= self.episode_steps - time
        # Calculate how many steps we need to take with action
        num_steps = jnp.minimum(steps_to_apply, self.episode_steps - time)

        # Integrate dynamics forward for the num_steps
        if self.time_as_part_of_state:
            old_obs = {
                'state': obs,
                'privileged_state': privileged_state_obs
            }
            integration_state = state.replace(obs=old_obs)
        else:
            integration_state = state.replace(pipeline_state=env_pipeline_state)

        def body_integration_step(val):
            s, r, i, current_accumulated_metrics_dict = val
            next_s = self.env.step(s, u)
            discount_factor = self.discounting ** i
            next_reward = r + discount_factor * (1 - next_s.done) * next_s.reward
            next_accumulated_metrics_dict = {}
            state_metrics = next_s.metrics
            for k, v in state_metrics.items():
                if k.startswith("reward"):
                    next_accumulated_metrics_dict[k] = current_accumulated_metrics_dict[k] + discount_factor * (
                                1 - next_s.done) * v
                elif k == 'swing_peak':
                    next_accumulated_metrics_dict[k] = jnp.where(next_s.done, current_accumulated_metrics_dict[k],
                                                                 jnp.maximum(current_accumulated_metrics_dict[k],v))
                else:
                    continue
            return next_s, next_reward, i + 1, next_accumulated_metrics_dict

        def cond_integration_step(val):
            s, r, i, metrics_dict = val
            # We continue if index is smaller that num_steps ant we are not done
            return jnp.bitwise_and(i < num_steps, jnp.bitwise_not(s.done.astype(bool)))

        initial_metrics_dict = {}
        for key in integration_state.metrics.keys():
            initial_metrics_dict[key] = jnp.array(0.0, dtype=jnp.float32)

        init_val = (integration_state, jnp.array(0.0), jnp.array(0), initial_metrics_dict)
        final_val = while_loop(cond_integration_step, body_integration_step, init_val)
        next_state, total_reward, index, final_metrics_dict = final_val
        next_done = 1 - (1 - next_state.done) * (1 - done)

        # Add switch cost to the total reward
        total_reward = total_reward - self.switch_cost(state=state.obs, action=u)

        # Prepare augmented obs (how many steps we actually took)
        next_time = (time + index)
        if self.time_as_part_of_state:
            augmented_next_obs = self._add_time_to_obs(next_state, next_time)
            augmented_next_state = next_state.replace(obs=augmented_next_obs,
                                                      reward=total_reward,
                                                      done=next_done,
                                                      metrics=final_metrics_dict)
            return augmented_next_state
        else:
            augmented_pipeline_state = AugmentedPipelineState(pipeline_state=next_state.pipeline_state,
                                                              time=next_time)
            augmented_next_state = next_state.replace(reward=total_reward,
                                                      done=next_done,
                                                      pipeline_state=augmented_pipeline_state,
                                                      metrics=final_metrics_dict)
            return augmented_next_state

    def simulation_step(self, state: mjx_env.State, action: jax.Array) -> (mjx_env.State, mjx_env.State):
        u, pseudo_time_for_action = action[:-1], action[-1]
        if self.time_as_part_of_state:
            obs, time, privileged_state_obs = self._get_time_and_obs(state)
        else:
            env_pipeline_state = state.pipeline_state.pipeline_state
            time = state.pipeline_state.time

        # Calculate the action time, i.e. Map pseudo_time_for_action from [-1, 1] to
        # time [self.min_time_between_switches, time_to_go] (now number of steps)
        steps_to_apply = self.compute_time(pseudo_time=pseudo_time_for_action,
                                           t_lower=self.min_time_between_switches,
                                           t_upper=self.max_time_between_switches)
        done = steps_to_apply >= self.episode_steps - time

        # Calculate how many steps we need to take with action
        num_steps = jnp.minimum(steps_to_apply, self.episode_steps - time)

        # Integrate dynamics forward for the num_steps

        if self.time_as_part_of_state:
            old_obs = {
                'state': obs,
                'privileged_state': privileged_state_obs
            }
            state = state.replace(obs=old_obs)
        else:
            state = state.replace(pipeline_state=env_pipeline_state)

        # Execute the action for the predicted number of integration steps
        step_index = 0
        cur_state = state
        all_states = []
        while step_index < num_steps and not cur_state.done:
            cur_state = self.jitted_step_fn(cur_state, u)
            all_states.append(cur_state)
            step_index += 1

        next_state = cur_state
        if len(all_states) == 0:
            all_states = [state]
        inner_part = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *all_states)
        total_reward = jnp.sum(inner_part.reward)
        next_done = 1 - (1 - next_state.done) * (1 - done)

        # Add switch cost to the total reward
        total_reward = total_reward - self.switch_cost(state=state.obs, action=u)

        # Prepare augmented obs
        next_time = (time + step_index)
        if self.time_as_part_of_state:
            augmented_next_obs = self._add_time_to_obs(next_state, next_time)
            augmented_next_state = next_state.replace(obs=augmented_next_obs,
                                                      reward=total_reward,
                                                      done=next_done)
            return augmented_next_state, inner_part
        else:
            augmented_pipeline_state = AugmentedPipelineState(pipeline_state=next_state.pipeline_state,
                                                              time=next_time.reshape())
            augmented_next_state = next_state.replace(reward=total_reward,
                                                      done=next_done,
                                                      pipeline_state=augmented_pipeline_state)
            return augmented_next_state, inner_part

    @property
    def backend(self) -> str:
        return self.env.backend

    @property
    def dt(self):
        return self.env.dt

    @property
    def observation_size(self) -> mjx_env.ObservationSize:
        if self.time_as_part_of_state:
            obs_size = {
                k: (v[0] + 1,) for k, v in self.env.observation_size.items()
            }
            return obs_size
        else:
            return self.env.observation_size

    @property
    def action_size(self) -> int:
        return self.env.action_size + 1

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def unwrapped(self) -> Any:
        return self._unwrapped
