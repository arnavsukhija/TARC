from typing import Dict, Tuple

import chex
import jax.numpy as jnp
from distrax import Distribution
from distrax import Normal
from mbpo.systems.base_systems import Reward

from envs.rccar.sim.rccar import RCCarEnvReward

@chex.dataclass
class CarRewardParams:
    _goal: chex.Array
    key: chex.PRNGKey


class CarReward(Reward[CarRewardParams]):
    _goal: jnp.array = jnp.array([0.0, 0.0, 0.0])

    def __init__(self,
                 ctrl_cost_weight: float = 0.005,
                 encode_angle: bool = False,
                 bound: float = 0.1,
                 margin_factor: float = 10.0,
                 num_frame_stack: int = 0,
                 ctrl_diff_weight: float = 0.0):
        Reward.__init__(self, x_dim=7 if encode_angle else 6, u_dim=2)
        self.num_frame_stack = num_frame_stack
        self.ctrl_diff_weight = ctrl_diff_weight
        self.ctrl_cost_weight = ctrl_cost_weight
        self.encode_angle: bool = encode_angle
        self._reward_model = RCCarEnvReward(goal=self._goal,
                                            ctrl_cost_weight=ctrl_cost_weight,
                                            encode_angle=self.encode_angle,
                                            bound=bound,
                                            margin_factor=margin_factor)

    def init_params(self, key: chex.PRNGKey) -> CarRewardParams:
        return CarRewardParams(_goal=self._goal, key=key)

    def set_goal(self, des_goal: chex.Array):
        self._goal = des_goal

    def __call__(self,
                 x: chex.Array,
                 u: chex.Array,
                 reward_params: CarRewardParams,
                 x_next: chex.Array | None = None) -> Tuple[Distribution, CarRewardParams]:
        assert x.shape == (self.x_dim + self.num_frame_stack * self.u_dim,) and u.shape == (self.u_dim,)
        assert x_next.shape == (self.x_dim + self.num_frame_stack * self.u_dim,)
        x_state, us_stacked = x[:self.x_dim], x[self.x_dim:]
        x_next_state, us_next_stacked = x_next[:self.x_dim], x_next[self.x_dim:]
        reward = self._reward_model.forward(obs=None, action=u, next_obs=x_next_state)
        # Now we add cost for the control inputs change:
        if self.num_frame_stack > 0:
            u_previous = us_stacked[-self.u_dim:]
            reward -= self.ctrl_diff_weight * jnp.sum((u_previous - u) ** 2)
        return Normal(reward, jnp.zeros_like(reward)), reward_params
