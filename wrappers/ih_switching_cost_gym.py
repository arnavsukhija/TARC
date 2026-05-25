from typing import NamedTuple, Tuple
import numpy as np
from gym import Env
from gym.spaces import Box

from envs.rccar.hardware.car_env_hardware import CarEnv


class AugmentedPipelineState(NamedTuple):
    pipeline_state: np.ndarray
    time: float


class SwitchCost:
    def __call__(self, state: np.ndarray, action: np.ndarray) -> float:
        """
        Computes the switch cost given state and value
        """
        raise NotImplementedError


class ConstantSwitchCost(SwitchCost):
    def __init__(self, value: float):
        self.value = value

    def __call__(self, state: np.ndarray, action: np.ndarray) -> float:
        return self.value


class IHSwitchCostWrapper(Env):
    """
    A gym env version of the SwitchCost Wrapper for wtc
    """
    def __init__(self,
                 env: CarEnv,
                 min_time_between_switches: float,
                 max_time_between_switches: float,
                 switch_cost: SwitchCost = ConstantSwitchCost(value=1.0),
                 discounting: float = 0.99,
                 time_as_part_of_state: bool = True, ):
        super().__init__()
        self.env = env
        self.switch_cost = switch_cost
        self.min_time_between_switches = min_time_between_switches # will be passed in seconds now
        self.max_time_between_switches = max_time_between_switches # will be passed in seconds now
        self.sim_dt = 1.0 / 30 # this is the dt we used in simulation for policy training
        assert time_as_part_of_state is True, \
            'Time should be part of the state, since this was used while training the policy' # we keep this as true since training was done using time as a state component
        assert min_time_between_switches >= self.sim_dt, \
            'Min steps between switches must be at least 1 / 30, this was the trained simulation dt' # in sim, this is the shortest action duration
        self.discounting = discounting
        self.time_as_part_of_state = time_as_part_of_state
        self.state = None # initialize state as none for now

    def reset(self, *args, **kwargs) -> Tuple[np.ndarray, dict]: # always set self.state to this
        """
        Resets the environment and return the initial augmented state
        """
        obs, info = self.env.reset(*args, **kwargs)  # Pass the args to CarEnv reset
        time = np.array(0.0, dtype=float)  # Initialize number of passed steps as 0

        if self.time_as_part_of_state:  # we augment the state by the time component
            augmented_obs = np.concatenate([obs, time.reshape(1)])
        else:  # we return the augmented pipeline state then
            augmented_obs = AugmentedPipelineState(pipeline_state=obs, time=0.0)
        self.state = augmented_obs
        return augmented_obs, info

    def compute_steps(self,
                      pseudo_time: float,
                      t_lower: float,
                      t_upper: float) -> int:
        time_for_action = ((t_upper - t_lower) / 2 * pseudo_time + (t_upper + t_lower) / 2) # returns the time in seconds between [tmin, tmax]
        return np.floor(time_for_action // self.sim_dt)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        u, pseudo_time_for_action = action[:-1], action[-1]

        if self.state is None:
            self.state, _ = self.reset()
        # we calculate the number of applied steps, map the according action from [-1,1] to [step_min, step_max]
        steps_for_action = self.compute_steps(pseudo_time = pseudo_time_for_action, t_lower=self.min_time_between_switches,
                                                     t_upper=self.max_time_between_switches)

        if self.time_as_part_of_state:
            obs, time = self.state[:-1], self.state[-1]  # time corresponds to the number of done steps * env.dt (so how much time has already passed)
        else:  # in this case, the state is an augmented pipeline state
            obs, time = self.state.pipeline_state, self.state.time

        num_steps = np.minimum(steps_for_action, self.env.max_steps - self.env.env_steps) # compute how many steps we can take

        total_reward = 0
        index = 0
        done = False # we perform the action for num_steps steps
        current_state = self.state

        # applying the number of steps in a loop
        while index < num_steps and not done:
            next_state, reward, done, info = self.env.step(
                action=u)  # each time we call step, the car system updates its env_steps, allowing for the car system to tell us if it has terminated
            total_reward += (self.discounting ** index) * (1 - done) * reward #in case the env terminates, the reward is not considered
            index += 1
            current_state = next_state

        total_reward = total_reward - self.switch_cost(state=self.state, action=u)

        # Augment state by time component
        new_time = (time + index * self.sim_dt).reshape(1) # updates the time we have had so far
        if self.time_as_part_of_state:
            augmented_next_state = np.concatenate([current_state, new_time])
        else:
            augmented_next_state = AugmentedPipelineState(pipeline_state=current_state, time=new_time)
        self.state = augmented_next_state  # update the state parameter
        return augmented_next_state, total_reward, done, {'time_elapsed': info['time_elapsed'],
                                                         'terminal_reward': info['terminal_reward']}


    def close(self):
        self.env.close()

    @property
    def observation_size(self) -> int:
        # +1 for time-to-go ant +1 for num remaining switches
        if self.time_as_part_of_state:
            return self.env.state_dim + 1
        else:
            return self.env.state_dim

    @property
    def action_size(self) -> int:
        # +1 for time that we apply action for
        return self.env.action_dim + 1

