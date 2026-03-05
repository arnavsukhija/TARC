import time

import gym
import sys

import jax.random
import numpy as np
from gym.spaces import Box
from envs.rccar.sim.rccar import encode_angles_numpy as encode_angles, decode_angles_numpy as decode_angles
from typing import Dict, Tuple, Any
from envs.rccar.hardware.car_system_reward import CarReward

X_MIN_LIMIT = -1.4
X_MAX_LIMIT = 2.8
Y_MIN_LIMIT = -2.6
Y_MAX_LIMIT = 1.5

class CarEnv(gym.Env):
    max_steps: int = 200
    _goal: np.array = np.array([0.0, 0.0, 0.0])
    _angle_idx: int = 2
    _init_state: np.array = np.array([1.42, -1.04, np.pi]) # np.array([1.4, -1.099, np.pi]) #

    def __init__(self,
                 car_id: int = 2,
                 control_time_ms: float = 24.,
                 max_wait_time: float = 1,
                 window_size: int = 6,
                 num_frame_stacks: int = 3,
                 port_number: int = 4,  # leftmost usb port in the display has port number 8
                 encode_angle: bool = True,
                 max_throttle: float = 0.4,
                 car_reward_kwargs: dict = None,
                 ):
        super().__init__()
        sys.path.append("C:/Users/Panda/Desktop/rcCarInterface/rc-car-interface/build/src/libs/pyCarController")

        assert 0.0 <= max_throttle <= 1.0
        self.max_throttle = max_throttle
        import carl
        self.control_frequency = 1 / (0.001 * control_time_ms)
        self.max_wait_time = max_wait_time
        self.window_size = window_size
        self.num_frame_stacks = num_frame_stacks
        self.port_number = port_number
        self.encode_angle = encode_angle
        if car_id == 1:
            mocap_id = 1003
        elif car_id == 2:
            mocap_id = 1034
        else:
            raise Exception("Only 2 cars have a mocap id")
        self.controller = carl.controller(w_size=window_size, p_number=port_number, mocap_id=mocap_id,
                                          wait_time=max_wait_time, control_freq=self.control_frequency)
        self.initial_reset = True
        self.controller_started = False

        self.num_frame_stacks = num_frame_stacks
        self.env_steps = 0

        # initialize reward model
        if car_reward_kwargs:
            self._reward_model = CarReward(**car_reward_kwargs, num_frame_stack=num_frame_stacks)
        else:
            self._reward_model = CarReward(num_frame_stack=num_frame_stacks, encode_angle=encode_angle)
        self._reward_model.set_goal(self._goal)
        self.reward_params = self._reward_model.init_params(jax.random.PRNGKey(0))

        # setup observation and action space
        high = np.ones(6 + self.encode_angle + 2 * num_frame_stacks) * np.inf
        if self.encode_angle:
            high[2:4] = 1
        high[6:] = 1
        self.observation_space = Box(low=-high, high=high, shape=(6 + self.encode_angle + 2 * num_frame_stacks,))
        self.action_space = Box(low=-1, high=1, shape=(2,))

        self.action_dim = 2
        self.state_dim = 7 if self.encode_angle else 6

        # init state
        self.state: np.array = np.zeros(shape=(self.state_dim,))
        self.stacked_last_actions: np.array = np.zeros(shape=(num_frame_stacks * self.action_dim))

    def log_mocap_info(self):
        logs = self.controller.get_mocap_logs()
        logs_dictionary = {
            'last_frame': logs.last_frame,
            'total_frames': logs.total_frames,
            'recorded_frames': logs.recorded_frames,
            'requested_frames': logs.requested_frames,
            'skipped_frames': logs.skipped_frames,
            'invalid_frames': logs.invalid_frames,
        }
        import os
        from datetime import datetime
        import csv
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        file_path = os.path.join(os.getcwd(), 'logs/')
        isExist = os.path.exists(file_path)
        if not isExist:
            os.makedirs(file_path)
        file_name = os.path.join(file_path, date_time + 's.csv')
        with open(file_name, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=logs_dictionary.keys())
            writer.writeheader()
            writer.writerow(logs_dictionary)

    def get_state_from_mocap(self):
        current_state = self.controller.get_state()
        mocap_x = current_state[[0, 3]]
        current_state[[0, 3]] = current_state[[1, 4]]
        current_state[[1, 4]] = mocap_x
        current_state[2] += np.pi
        current_state = self.normalize_theta_in_state(current_state)
        return current_state

    def reset(self, *args, **kwargs):
        if not self.initial_reset:
            self.log_mocap_info()
        self.initial_reset = False
        self.env_steps = 0

        # dialogue with user
        answer = input("Press Y to continue the reset.")
        assert answer == 'Y' or answer == 'y', "environment execution aborted."
        if not self.controller_started:
            self.controller.start()
            print("Starting controller in ~3 sec")
            time.sleep(3)
            self.controller_started = True

        # init last actions to zeros
        self.stacked_last_actions = np.zeros(2 * self.num_frame_stacks)

        # get current state
        state = self.get_state_from_mocap()
        state[0:3] = state[0:3] - self._goal
        if self.encode_angle:
            state = encode_angles(state, angle_idx=self._angle_idx)
        self.state = state

        state_with_last_acts = np.concatenate([self.state, self.stacked_last_actions], axis=-1)
        assert self.observation_space.shape == state_with_last_acts.shape
        return state_with_last_acts, {}

    def close(self):
        self.controller.stop()
        self.log_mocap_info()
        self.controller_started = False

    def step(self, action: np.array) -> Tuple[np.array, float, bool, Dict[str, Any]]:
        """ Performs one step on the real car (sends commands to)

        Args:
            action: numpy array of shape (2,) with [steer, throttle]
        """

        # update time
        self.env_steps += 1
        # check, clip and rescale actions
        assert np.shape(action) == (2,)
        steer, throttle = action

        action = np.clip([steer, throttle], -1.0, 1.0)
        scaled_action = action.copy()
        scaled_action[1] *= self.max_throttle


        # set action command on car and wait for some time
        self.controller.control_mode()  # sets the mode to control
        command_set_in_time = self.controller.set_command(scaled_action)  # set action
        assert command_set_in_time, "API blocked python thread for too long"

        time_elapsed = self.controller.get_time_elapsed()  # time elapsed since last action. this should be duration after waiting as well

        # keep last state
        _last_state = self.state
        _last_state_with_last_acts = np.concatenate([_last_state, self.stacked_last_actions], axis=-1)

        # get current state
        raw_state = self.get_state_from_mocap()
        state = raw_state
        state[0:3] = state[0:3] - self._goal
        if self.encode_angle:
            state = encode_angles(state, angle_idx=self._angle_idx)
            raw_state = encode_angles(raw_state, angle_idx=self._angle_idx)
        self.state = state

        # take care of frame stacking
        if self.num_frame_stacks > 0:
            # throw out oldest action and add latest action
            self.stacked_last_actions = np.concatenate([self.stacked_last_actions[2:], action])
        assert self.stacked_last_actions.shape == (self.action_dim * self.num_frame_stacks,)

        # add last actions to state
        state_with_last_acts = np.concatenate([self.state, self.stacked_last_actions], axis=-1)
        assert self.observation_space.shape == state_with_last_acts.shape

        # compute reward
        reward = self.reward(_last_state_with_last_acts, action, state_with_last_acts)

        # check termination conditions
        terminate, terminal_reward = self.terminate(raw_state)
        return state_with_last_acts, reward, terminate, {'time_elapsed': time_elapsed,
                                                         'terminal_reward': terminal_reward}

    def reward(self, last_state, action, state):
        dist, self.reward_params = self._reward_model(
            x=last_state,
            u=action,
            reward_params=self.reward_params,
            x_next=state,
        )
        return dist.mean()

    def terminate(self, state: np.array):
        reached_goal = self.reached_goal(state, self._goal)
        out_of_bound = self.constraint_violation(state)
        time_out = self.env_steps >= self.max_steps #we have reached the time_out now, too many steps done

        if reached_goal:
            print("REACHED GOAL!")
        elif out_of_bound:
            print("CONSTRAINT VIOLATION!")
        elif time_out:
            print("TIMEOUT!")
        terminate = reached_goal or out_of_bound or time_out
        terminal_reward = 0.0
        if reached_goal:
            terminal_reward += self.max_steps - self.env_steps #so the steps that still remain
        return terminate, terminal_reward

    def reached_goal(self, state: np.array, goal: np.array) -> bool:
        if self.encode_angle:
            state = decode_angles(state, angle_idx=self._angle_idx)
        assert state.shape == (6,)
        assert goal.shape == (3,)

        dist = np.sqrt(np.square(state[:2] - goal[:2]).sum(-1))
        ang_dev = np.abs(self.normalize_theta(state[2] - goal[2]))
        speed = np.sqrt(np.square(state[..., 3:]).sum(-1))

        in_bounds = np.logical_and(np.logical_and(dist < 0.05, ang_dev < 0.2), speed < 0.5)
        return in_bounds

    @staticmethod
    def constraint_violation(state: np.array) -> bool:
        in_bounds = np.logical_and(
            np.logical_and(X_MIN_LIMIT <= state[0], state[0] <= X_MAX_LIMIT),
            np.logical_and(Y_MIN_LIMIT <= state[1], state[1] <= Y_MAX_LIMIT)
        )
        return not in_bounds

    @staticmethod
    def normalize_theta(theta: np.array) -> np.ndarray:
        theta = ((theta + np.pi) % (2 * np.pi)) - np.pi
        return theta


    def normalize_theta_in_state(self, state: np.array) -> np.array:
        state[2] = self.normalize_theta(state[2])
        return state

    def reset_to_origin(self):
        def convert_obs(env_obs):
            obs = decode_angles(env_obs, angle_idx=self._angle_idx)
            obs[0:3] = obs[0:3] - self._init_state[0:3]
            obs[0:2] *= -1
            obs[3:5] *= -1
            return encode_angles(obs, angle_idx=self._angle_idx)

        obs = np.concatenate([self.state, self.stacked_last_actions], axis=-1)
        for i in range(200):
            action = np.clip(np.asarray(self.reset_policy(obs)), -0.6, 0.6)
            obs, reward, terminate, info = self.step(action)
            obs = convert_obs(obs)
            if terminate:
                break

        time.sleep(5)
