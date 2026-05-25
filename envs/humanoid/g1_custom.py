import jax.numpy as jp
from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick


class G1JoystickCustom(g1_joystick.Joystick):
    """G1 joystick env with weighted pose cost."""

    def _cost_pose(self, qpos: jp.ndarray) -> jp.ndarray:
        return jp.sum(jp.square(qpos - self._default_pose) * self._weights)
