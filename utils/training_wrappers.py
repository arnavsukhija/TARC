import jax
from brax.envs.base import Wrapper
from jax import numpy as jp

class DomainRandomizationVmapBase(Wrapper):
    """Base class for domain randomization wrappers."""

    def __init__(self, env, randomization_fn, *, augment_state=True):
        super().__init__(env)
        self.augment_state = augment_state
        (
            self._randomized_models,
            self._in_axes,
            self.domain_parameters,
        ) = self._init_randomization(randomization_fn)
        dummy = self.env.reset(jax.random.PRNGKey(0))
        self.strip_privileged_state = isinstance(dummy.obs, jax.Array)

    def _init_randomization(self, randomization_fn):
        """To be implemented by subclasses to handle model-specific randomization."""
        raise NotImplementedError

    def _env_fn(self, model):
        """To be implemented by subclasses to return an environment with the given model."""
        raise NotImplementedError

    def reset(self, rng: jax.Array):
        def reset_fn(model, rng):
            env = self._env_fn(model)
            return env.reset(rng)

        state = jax.vmap(reset_fn, in_axes=[self._in_axes, 0])(
            self._randomized_models, rng
        )
        if self.augment_state:
            state = self._add_privileged_state(state)
        return state

    def step(self, state, action: jax.Array):
        def step_fn(model, s, a):
            env = self._env_fn(model)
            return env.step(s, a)

        if self.augment_state and self.strip_privileged_state:
            state = state.replace(obs=state.obs["state"])

        state = jax.vmap(step_fn, in_axes=[self._in_axes, 0, 0])(
            self._randomized_models, state, action
        )
        if self.augment_state:
            state = self._add_privileged_state(state)
        return state

    def _add_privileged_state(self, state):
        """Adds privileged state to the observation if augmentation is enabled."""
        if isinstance(state.obs, jax.Array):
            state = state.replace(
                obs={
                    "state": state.obs,
                    "privileged_state": jp.concatenate(
                        [state.obs, self.domain_parameters], -1
                    ),
                }
            )
        else:
            state = state.replace(
                obs={
                    "state": state.obs["state"],
                    "privileged_state": jp.concatenate(
                        [state.obs["privileged_state"], self.domain_parameters], -1
                    ),
                }
            )
        return state

    @property
    def observation_size(self):
        """Compute observation size based on the augmentation setting."""
        if not self.augment_state:
            return self.env.observation_size

        if isinstance(self.env.observation_size, int):
            return {
                "state": (self.env.observation_size,),
                "privileged_state": (
                    self.env.observation_size + self.domain_parameters.shape[1],
                ),
            }
        else:
            return {
                "state": (self.env.observation_size["state"],),
                "privileged_state": (
                    self.env.observation_size["privileged_state"]
                    + self.domain_parameters.shape[1],
                ),
            }


class DomainRandomizationVmapWrapper(DomainRandomizationVmapBase):
    def _init_randomization(self, randomization_fn):
        return randomization_fn(self.sys)

    def _env_fn(self, model):
        env = self.env
        env.unwrapped._dynamics_params = model
        return env