import jax.numpy as jnp
import chex


def discrete_to_continuous_discounting(discrete_discounting: chex.Array, dt: chex.Array) -> chex.Array:
    return jnp.log(1 / discrete_discounting) / dt


def continuous_to_discrete_discounting(continuous_discounting: chex.Array, dt: chex.Array) -> chex.Array:
    return jnp.exp(-continuous_discounting * dt)
