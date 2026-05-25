import jax
import jax.numpy as jnp
from brax.envs.base import State, Env, Wrapper


class ActionDelayWrapper(Wrapper):
    """
    Brax wrapper that adds an action-delay to the base environment (a transformation of the underlying MDP to a new augmented MDP)
    """
    def __init__(self, env: Env, action_delay: float, ctrl_diff_weight: float = 0.0):
        super().__init__(env)
        self.ctrl_diff_weight = ctrl_diff_weight

        # Buffer parameters for action delay buffer
        self.dt = env.dt
        assert(action_delay >= 1)
        self.buffer_size = int(jnp.ceil(action_delay)) + 1
        if action_delay % 1 == 0:  # If action_delay is an integer, no interpolation
            self.interp_weights = jnp.array([1.0, 0.0])
        else:  # Otherwise, compute fractional interpolation weights
            weight_on_first = action_delay % 1
            self.interp_weights = jnp.array([weight_on_first, 1.0 - weight_on_first])

    def reset(self, rng: jax.Array) -> State:
        "Uses the environments reset function and adds the action_delay buffer to the new state as part of info (easy maintenance), to maintain Markovian property"
        state = self.env.reset(rng)
        action_buffer = jnp.zeros((self.buffer_size, self.action_size))
        new_obs = jnp.concatenate([state.obs, action_buffer.flatten()])
        return state.replace(obs=new_obs)

    def step(self, state: State, action: jax.Array) -> State:
        """We take a step with delayed action"""
        # get delayed action (interpolate between two actions if the delay is not a multiple of dt)
        obs, action_buffer = state.obs[:self.env.observation_size], state.obs[self.env.observation_size:].reshape(self.buffer_size, self.action_size)
        # we reset the original structure of the state so that the base environment can process it easily
        # Debugging inside step function
        delayed_action = jnp.sum(action_buffer[:2] * self.interp_weights[:, None], axis=0)
        state = state.replace(obs=obs) # we restore the original state structure
        next_state = self.env.step(state, delayed_action)
        # we derive the new action buffer and pass it accordingly
        new_action_buffer = jnp.concatenate([action_buffer[1:], action[None]], axis=0)
        new_obs = jnp.concatenate([next_state.obs, new_action_buffer.flatten()])
        control_penalty = -self.ctrl_diff_weight * jnp.sum((action - action_buffer[-1]) ** 2)  #compute control penalty based on the predicted action and the last action in buffer
        return next_state.replace(obs=new_obs, reward = next_state.reward + control_penalty)

    @property
    def observation_size(self) -> int:
        return self.env.observation_size + (self.buffer_size * self.action_size) #size after flattening

    @property
    def action_size(self) -> int:
        return self.env.action_size
