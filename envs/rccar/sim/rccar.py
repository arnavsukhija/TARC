from abc import ABC, abstractmethod
from functools import partial
from typing import Tuple, NamedTuple, Union, Optional, Dict, List
import jax
import jax.numpy as jnp
import jax.flatten_util
import jax.tree_util as jtu
import numpy as np
from brax.envs.base import State, Env
from flax import struct
from jaxtyping import PyTree

from utils.tolerance_reward import ToleranceReward

OBS_NOISE_STD_SIM_CAR: jnp.array = 0.1 * jnp.exp(jnp.array([-4.5, -4.5, -4., -2.5, -2.5, -1.]))

X_LIM = (-0.3, 3.0)
Y_LIM = (-2.1, 1.5)


def rotate_coordinates(state: jnp.array, encode_angle: bool = False) -> jnp.array:
    x_pos, x_vel = state[..., 0:1], state[..., 3 + int(encode_angle): 4 + int(encode_angle)]
    y_pos, y_vel = state[..., 1:2], state[:, 4 + int(encode_angle):5 + int(encode_angle)]
    theta = state[..., 2: 3 + int(encode_angle)]
    new_state = jnp.concatenate([y_pos, -x_pos, theta, y_vel, -x_vel, state[..., 5 + int(encode_angle):]],
                                axis=-1)
    assert state.shape == new_state.shape
    return new_state


def plot_rc_trajectory(traj: jnp.array, actions: Optional[jnp.array] = None, pos_domain_size: float = 5,
                       show: bool = True, encode_angle: bool = False):
    """ Plots the trajectory of the RC car """
    if encode_angle:
        traj = decode_angles(traj, 2)

    import matplotlib.pyplot as plt
    scale_factor = 1.5
    if actions is None:
        fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(scale_factor * 12, scale_factor * 8))
    else:
        fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(scale_factor * 16, scale_factor * 8))
    axes[0][0].set_xlim(-pos_domain_size, pos_domain_size)
    axes[0][0].set_ylim(-pos_domain_size, pos_domain_size)
    axes[0][0].scatter(0, 0)
    # axes[0][0].plot(traj[:, 0], traj[:, 1])
    axes[0][0].set_title('x-y')

    # chaange x -> -y and y -> x
    traj = rotate_coordinates(traj, encode_angle=False)

    # Plot the velocity of the car as vectors
    total_vel = jnp.sqrt(traj[:, 3] ** 2 + traj[:, 4] ** 2)
    axes[0][0].quiver(traj[0:-1:3, 0], traj[0:-1:3, 1], traj[0:-1:3, 3], traj[0:-1:3, 4],
                      total_vel[0:-1:3], cmap='jet', scale=20,
                      headlength=2, headaxislength=2, headwidth=2, linewidth=0.2)

    t = jnp.arange(traj.shape[0]) / 30.
    # theta
    axes[0][1].plot(t, traj[:, 2])
    axes[0][1].set_xlabel('time')
    axes[0][1].set_ylabel('theta')
    axes[0][1].set_title('theta')

    # angular velocity
    axes[0][2].plot(t, traj[:, -1])
    axes[0][2].set_xlabel('time')
    axes[0][2].set_ylabel('angular velocity')
    axes[0][2].set_title('angular velocity')

    axes[1][0].plot(t, total_vel)
    axes[1][0].set_xlabel('time')
    axes[1][0].set_ylabel('total velocity')
    axes[1][0].set_title('velocity')

    # vel x
    axes[1][1].plot(t, traj[:, 3])
    axes[1][1].set_xlabel('time')
    axes[1][1].set_ylabel('velocity x')
    axes[1][1].set_title('velocity x')

    axes[1][2].plot(t, traj[:, 4])
    axes[1][2].set_xlabel('time')
    axes[1][2].set_ylabel('velocity y')
    axes[1][2].set_title('velocity y')

    if actions is not None:
        # steering
        axes[0][3].plot(t[:actions.shape[0]], actions[:, 0])
        axes[0][3].set_xlabel('time')
        axes[0][3].set_ylabel('steer')
        axes[0][3].set_title('steering')

        # throttle
        axes[1][3].plot(t[:actions.shape[0]], actions[:, 1])
        axes[1][3].set_xlabel('time')
        axes[1][3].set_ylabel('throttle')
        axes[1][3].set_title('throttle')


    fig.tight_layout()
    if show:
        fig.show()
    return fig, axes

def encode_angles(state: jnp.array, angle_idx: int) -> jnp.array:
    """ Encodes the angle (theta) as sin(theta) ant cos(theta) """
    assert angle_idx <= state.shape[-1] - 1
    theta = state[..., angle_idx:angle_idx + 1]
    state_encoded = jnp.concatenate([state[..., :angle_idx], jnp.sin(theta), jnp.cos(theta),
                                     state[..., angle_idx + 1:]], axis=-1)
    assert state_encoded.shape[-1] == state.shape[-1] + 1
    return state_encoded

def encode_angles_numpy(state: np.array, angle_idx: int) -> np.array:
    """ Encodes the angle (theta) as sin(theta) and cos(theta) """
    assert angle_idx <= state.shape[-1] - 1
    theta = state[..., angle_idx:angle_idx + 1]
    state_encoded = np.concatenate([state[..., :angle_idx], np.sin(theta), np.cos(theta),
                                     state[..., angle_idx + 1:]], axis=-1)
    assert state_encoded.shape[-1] == state.shape[-1] + 1
    return state_encoded

def decode_angles(state: jnp.array, angle_idx: int) -> jnp.array:
    """ Decodes the angle (theta) from sin(theta) ant cos(theta)"""
    assert angle_idx < state.shape[-1] - 1
    theta = jnp.arctan2(state[..., angle_idx:angle_idx + 1],
                        state[..., angle_idx + 1:angle_idx + 2])
    state_decoded = jnp.concatenate([state[..., :angle_idx], theta, state[..., angle_idx + 2:]], axis=-1)
    assert state_decoded.shape[-1] == state.shape[-1] - 1
    return state_decoded

def decode_angles_numpy(state: np.array, angle_idx: int) -> np.array:
    """ Decodes the angle (theta) from sin(theta) and cos(theta)"""
    assert angle_idx < state.shape[-1] - 1
    theta = np.arctan2(state[..., angle_idx:angle_idx + 1],
                        state[..., angle_idx + 1:angle_idx + 2])
    state_decoded = np.concatenate([state[..., :angle_idx], theta, state[..., angle_idx + 2:]], axis=-1)
    assert state_decoded.shape[-1] == state.shape[-1] - 1
    return state_decoded


class CarParams(NamedTuple):
    """
    d_f, d_r : Represent grip of the car. Range: [0.015, 0.025]
    b_f, b_r: Slope of the pacejka. Range: [2.0 - 4.0].

    delta_limit: [0.3 - 0.5] -> Limit of the steering angle.

    c_m_1: Motor parameter. Range [0.2, 0.5]
    c_m_1: Motor friction, Range [0.00, 0.007]
    c_f, c_r: [1.0 2.0] # motor parameters: source https://web.mit.edu/drela/Public/web/qprop/motor1_theory.pdf,
    https://ethz.ch/content/dam/ethz/special-interest/mavt/dynamic-systems-n-control/idsc-dam/Lectures/Embedded
    -Control-Systems/LectureNotes/6_Motor_Control.pdf # or look at:
    https://video.ethz.ch/lectures/d-mavt/2021/spring/151-0593-00L/00718f4f-116b-4645-91da-b9482164a3c7.html :
    lecture 2 part 2
    c_m_1: max current of motor: [0.2 - 0.5] c_m_2: motor resistance due to shaft: [0.01 - 0.15]
    """
    m: Union[jax.Array, float] = jnp.array(1.65)  # [0.04, 0.08]
    i_com: Union[jax.Array, float] = jnp.array(2.78e-05)  # [1e-6, 5e-6]
    l_f: Union[jax.Array, float] = jnp.array(0.13)  # [0.025, 0.05]
    l_r: Union[jax.Array, float] = jnp.array(0.17)  # [0.025, 0.05]
    g: Union[jax.Array, float] = jnp.array(9.81)

    d_f: Union[jax.Array, float] = jnp.array(0.02)  # [0.015, 0.025]
    c_f: Union[jax.Array, float] = jnp.array(1.2)  # [1.0, 2.0]
    b_f: Union[jax.Array, float] = jnp.array(2.58)  # [2.0, 4.0]

    d_r: Union[jax.Array, float] = jnp.array(0.017)  # [0.015, 0.025]
    c_r: Union[jax.Array, float] = jnp.array(1.27)  # [1.0, 2.0]
    b_r: Union[jax.Array, float] = jnp.array(3.39)  # [2.0, 4.0]

    c_m_1: Union[jax.Array, float] = jnp.array(10.431917)  # [0.2, 0.5]
    c_m_2: Union[jax.Array, float] = jnp.array(1.5003588)  # [0.00, 0.007]
    c_d: Union[jax.Array, float] = jnp.array(0.0)  # [0.01, 0.1]
    steering_limit: Union[jax.Array, float] = jnp.array(0.19989373)
    use_blend: Union[jax.Array, float] = jnp.array(0.0)  # 0.0 -> (only kinematics), 1.0 -> (kinematics + dynamics)

    # parameters used to compute the blend ratio characteristics
    blend_ratio_ub: Union[jax.Array, float] = jnp.array([0.5477225575])
    blend_ratio_lb: Union[jax.Array, float] = jnp.array([0.4472135955])
    angle_offset: Union[jax.Array, float] = jnp.array([0.02791893])

@struct.dataclass
class CarPipelineState:
  """Pipeline state to store params of the simulator"""
  car_params: CarParams
class DynamicsModel(ABC):
    def __init__(self,
                 dt: float,
                 x_dim: int,
                 u_dim: int,
                 params: PyTree,
                 angle_idx: Optional[Union[int, jax.Array]] = None,
                 dt_integration: float = 1/90,
                 ):
        self.dt = dt
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.params = params
        self.angle_idx = angle_idx

        self.dt_integration = dt_integration
        assert dt >= dt_integration
        assert round(dt / dt_integration) == dt / dt_integration, 'dt must be a multiple of dt_integration'
        self._num_steps_integrate = int(dt / dt_integration)

    def next_step(self, x: jax.Array, u: jax.Array, params: PyTree) -> jax.Array:
        def body(carry, _):
            q = carry + self.dt_integration * self.ode(carry, u, params)
            return q, None

        next_state, _ = jax.lax.scan(body, x, xs=None, length=self._num_steps_integrate)
        if self.angle_idx is not None:
            theta = next_state[self.angle_idx]
            sin_theta, cos_theta = jnp.sin(theta), jnp.cos(theta)
            next_state = next_state.at[self.angle_idx].set(jnp.arctan2(sin_theta, cos_theta))
        return next_state

    def ode(self, x: jax.Array, u: jax.Array, params) -> jax.Array:
        assert x.shape[-1] == self.x_dim and u.shape[-1] == self.u_dim
        return self._ode(x, u, params)

    @abstractmethod
    def _ode(self, x: jax.Array, u: jax.Array, params) -> jax.Array:
        pass

    def _split_key_like_tree(self, key: jax.random.PRNGKey):
        treedef = jtu.tree_structure(self.params)
        keys = jax.random.split(key, treedef.num_leaves)
        return jtu.tree_unflatten(treedef, keys)

    def sample_params_uniform(self, key: jax.random.PRNGKey,
                              lower_bound: NamedTuple, upper_bound: NamedTuple, sample_shape: Optional[Union[int, Tuple[int]]] = None):
        keys = self._split_key_like_tree(key)
        if sample_shape is None:
            return jtu.tree_map(
                lambda key, l, u: jax.random.uniform(key, shape=l.shape, minval=l, maxval=u),
                keys, lower_bound, upper_bound)
        else:
            if isinstance(sample_shape, int):
                sample_shape = (sample_shape,)
            return jtu.tree_map(lambda key, l, u: jax.random.uniform(key, shape=sample_shape + l.shape, minval=l, maxval=u),
                                keys, lower_bound, upper_bound)


class RaceCar(DynamicsModel):
    """
    local_coordinates: bool
        Used to indicate if local or global coordinates shall be used.
        If local, the state x is
            x = [0, 0, theta, vel_r, vel_t, angular_velocity_z]
        else:
            x = [x, y, theta, vel_x, vel_y, angular_velocity_z]
    u = [steering_angle, throttle]
    encode_angle: bool
        Encodes angle to sin ant cos if true
    """

    def __init__(self, dt, encode_angle: bool = True, local_coordinates: bool = False, rk_integrator: bool = True):
        self.encode_angle = encode_angle
        x_dim = 6
        if dt <= 1 / 90:
            integration_dt = dt
        else:
            integration_dt = 1 / 90
        super().__init__(dt=dt, x_dim=x_dim, u_dim=2, params=CarParams(), angle_idx=2,
                         dt_integration=integration_dt)
        self.local_coordinates = local_coordinates
        self.angle_idx = 2
        self.velocity_start_idx = 4 if self.encode_angle else 3
        self.velocity_end_idx = 5 if self.encode_angle else 4
        self.rk_integrator = rk_integrator

    def rk_integration(self, x: jnp.array, u: jnp.array, params: CarParams) -> jnp.array:
        integration_factors = jnp.asarray([self.dt_integration / 2.,
                                           self.dt_integration / 2., self.dt_integration,
                                           self.dt_integration])
        integration_weights = jnp.asarray([self.dt_integration / 6.,
                                           self.dt_integration / 3., self.dt_integration / 3.0,
                                           self.dt_integration / 6.0])

        def body(carry, _):
            """one step of rk integration.
            k_0 = self.ode(x, u)
            k_1 = self.ode(x + self.dt_integration / 2. * k_0, u)
            k_2 = self.ode(x + self.dt_integration / 2. * k_1, u)
            k_3 = self.ode(x + self.dt_integration * k_2, u)

            x_next = x + self.dt_integration * (k_0 / 6. + k_1 / 3. + k_2 / 3. + k_3 / 6.)
            """

            def rk_integrate(carry, ins):
                k = self.ode(carry, u, params)
                carry = carry + k * ins
                outs = k
                return carry, outs

            _, dxs = jax.lax.scan(rk_integrate, carry, xs=integration_factors, length=4)
            dx = (dxs.T * integration_weights).sum(axis=-1)
            q = carry + dx
            return q, None

        next_state, _ = jax.lax.scan(body, x, xs=None, length=self._num_steps_integrate)
        if self.angle_idx is not None:
            theta = next_state[self.angle_idx]
            sin_theta, cos_theta = jnp.sin(theta), jnp.cos(theta)
            next_state = next_state.at[self.angle_idx].set(jnp.arctan2(sin_theta, cos_theta))
        return next_state

    def next_step(self, x: jnp.array, u: jnp.array, params: CarParams) -> jnp.array:
        theta_x = jnp.arctan2(x[..., self.angle_idx], x[..., self.angle_idx + 1]) if self.encode_angle else \
            x[..., self.angle_idx]
        offset = jnp.clip(params.angle_offset, -jnp.pi, jnp.pi)
        theta_x = theta_x + offset
        if not self.local_coordinates:
            # rotate velocity to local frame to compute dx
            velocity_global = x[..., self.velocity_start_idx: self.velocity_end_idx + 1]
            rotated_vel = self.rotate_vector(velocity_global,
                                             -theta_x)
            x = x.at[..., self.velocity_start_idx: self.velocity_end_idx + 1].set(rotated_vel)
        if self.encode_angle:
            x_reduced = self.reduce_x(x)
            if self.rk_integrator:
                x_reduced = self.rk_integration(x_reduced, u, params)
            else:
                x_reduced = super().next_step(x_reduced, u, params)
            next_theta = jnp.atleast_1d(x_reduced[..., self.angle_idx])
            next_x = jnp.concatenate([x_reduced[..., 0:self.angle_idx], jnp.sin(next_theta), jnp.cos(next_theta),
                                      x_reduced[..., self.angle_idx + 1:]], axis=-1)
        else:
            if self.rk_integrator:
                next_x = self.rk_integration(x, u, params)
            else:
                next_x = super().next_step(x, u, params)

        if self.local_coordinates:
            # convert position to local frame
            pos = next_x[..., 0:self.angle_idx] - x[..., 0:self.angle_idx]
            rotated_pos = self.rotate_vector(pos, -theta_x)
            next_x = next_x.at[..., 0:self.angle_idx].set(rotated_pos)
        else:
            # convert velocity to global frame
            new_theta_x = jnp.arctan2(next_x[..., self.angle_idx], next_x[..., self.angle_idx + 1]) \
                if self.encode_angle else next_x[..., self.angle_idx]
            new_theta_x = new_theta_x + offset
            velocity = next_x[..., self.velocity_start_idx: self.velocity_end_idx + 1]
            rotated_vel = self.rotate_vector(velocity, new_theta_x)
            next_x = next_x.at[..., self.velocity_start_idx: self.velocity_end_idx + 1].set(rotated_vel)

        return next_x

    def reduce_x(self, x):
        theta = jnp.arctan2(x[..., self.angle_idx], x[..., self.angle_idx + 1])

        x_reduced = jnp.concatenate([x[..., 0:self.angle_idx], jnp.atleast_1d(theta),
                                     x[..., self.velocity_start_idx:]],
                                    axis=-1)
        return x_reduced

    @staticmethod
    def rotate_vector(v, theta):
        v_x, v_y = v[..., 0], v[..., 1]
        rot_x = v_x * jnp.cos(theta) - v_y * jnp.sin(theta)
        rot_y = v_x * jnp.sin(theta) + v_y * jnp.cos(theta)
        return jnp.concatenate([jnp.atleast_1d(rot_x), jnp.atleast_1d(rot_y)], axis=-1)

    def _accelerations(self, x, u, params: CarParams):
        """Compute acceleration forces for dynamic model.
        Inputs
        -------
        x: jnp.ndarray,
            shape = (6, ) -> [x, y, theta, velocity_r, velocity_t, angular_velocity_z]
        u: jnp.ndarray,
            shape = (2, ) -> [steering_angle, throttle]

        Output
        ------
        acceleration: jnp.ndarray,
            shape = (3, ) -> [a_r, a_t, a_theta]
        """
        i_com = params.i_com
        theta, v_x, v_y, w = x[2], x[3], x[4], x[5]
        m = params.m
        l_f = params.l_f
        l_r = params.l_r
        d_f = params.d_f * params.g
        d_r = params.d_r * params.g
        c_f = params.c_f
        c_r = params.c_r
        b_f = params.b_f
        b_r = params.b_r
        c_m_1 = params.c_m_1
        c_m_2 = params.c_m_2

        c_d = params.c_d

        delta, d = u[0], u[1]

        alpha_f = - jnp.arctan(
            (w * l_f + v_y) /
            (v_x + 1e-6)
        ) + delta
        alpha_r = jnp.arctan(
            (w * l_r - v_y) /
            (v_x + 1e-6)
        )
        f_f_y = d_f * jnp.sin(c_f * jnp.arctan(b_f * alpha_f))
        f_r_y = d_r * jnp.sin(c_r * jnp.arctan(b_r * alpha_r))
        f_r_x = (c_m_1 * d - (c_m_2 ** 2) * v_x - (c_d ** 2) * (v_x * jnp.abs(v_x)))

        v_x_dot = (f_r_x - f_f_y * jnp.sin(delta) + m * v_y * w) / m
        v_y_dot = (f_r_y + f_f_y * jnp.cos(delta) - m * v_x * w) / m
        w_dot = (f_f_y * l_f * jnp.cos(delta) - f_r_y * l_r) / i_com

        acceleration = jnp.array([v_x_dot, v_y_dot, w_dot])
        return acceleration

    def _ode_dyn(self, x, u, params: CarParams):
        """Compute derivative using dynamic model.
        Inputs
        -------
        x: jnp.ndarray,
            shape = (6, ) -> [x, y, theta, velocity_r, velocity_t, angular_velocity_z]
        u: jnp.ndarray,
            shape = (2, ) -> [steering_angle, throttle]

        Output
        ------
        x_dot: jnp.ndarray,
            shape = (6, ) -> time derivative of x

        """
        # state = [p_x, p_y, theta, v_x, v_y, w]. Velocities are in local coordinate frame.
        # Inputs: [\delta, d] -> \delta steering angle ant d duty cycle of the electric motor.
        theta, v_x, v_y, w = x[2], x[3], x[4], x[5]
        p_x_dot = v_x * jnp.cos(theta) - v_y * jnp.sin(theta)
        p_y_dot = v_x * jnp.sin(theta) + v_y * jnp.cos(theta)
        theta_dot = w
        p_x_dot = jnp.array([p_x_dot, p_y_dot, theta_dot])

        accelerations = self._accelerations(x, u, params)

        x_dot = jnp.concatenate([p_x_dot, accelerations], axis=-1)
        return x_dot

    def _compute_dx_kin(self, x, u, params: CarParams):
        """Compute kinematics derivative for localized state.
        Inputs
        -----
        x: jnp.ndarray,
            shape = (6, ) -> [x, y, theta, v_x, v_y, w], velocities in local frame
        u: jnp.ndarray,
            shape = (2, ) -> [steering_angle, throttle]

        Output
        ------
        dx_kin: jnp.ndarray,
            shape = (6, ) -> derivative of x

        Assumption: \dot{\delta} = 0.
        """
        p_x, p_y, theta, v_x, v_y, w = x[0], x[1], x[2], x[3], x[4], x[5]  # progress
        m = params.m
        l_f = params.l_f
        l_r = params.l_r
        c_m_1 = params.c_m_1
        c_m_2 = params.c_m_2
        c_d = params.c_d
        delta, d = u[0], u[1]
        v_r = v_x
        v_r_dot = (c_m_1 * d - (c_m_2 ** 2) * v_r - (c_d ** 2) * (v_r * jnp.abs(v_r))) / m
        beta = jnp.arctan(jnp.tan(delta) * 1 / (l_r + l_f))
        v_x_dot = v_r_dot * jnp.cos(beta)
        # Determine accelerations from the kinematic model using FD.
        v_y_dot = (v_r * jnp.sin(beta) * l_r - v_y) / self.dt_integration
        # v_x_dot = (v_r_dot + v_y * w)
        # v_y_dot = - v_x * w
        w_dot = (jnp.sin(beta) * v_r - w) / self.dt_integration
        p_g_x_dot = v_x * jnp.cos(theta) - v_y * jnp.sin(theta)
        p_g_y_dot = v_x * jnp.sin(theta) + v_y * jnp.cos(theta)
        dx_kin = jnp.asarray([p_g_x_dot, p_g_y_dot, w, v_x_dot, v_y_dot, w_dot])
        return dx_kin

    def _compute_dx(self, x, u, params: CarParams):
        """Calculate time derivative of state.
        Inputs:
        ------
        x: jnp.ndarray,
            shape = (6, ) -> [x, y, theta, vel_r, vel_t, angular_velocity_z]
        u: jnp.ndarray,
            shape = (2, ) -> [steering_angle, throttle]
        params: CarParams,

        Output:
        -------
        dx: jnp.ndarray, derivative of x


        If params.use_blend <= 0.5 --> only kinematic model is used, else a blend between nonlinear model
        ant kinematic is used.
        """
        use_kin = params.use_blend <= 0.5
        v_x = x[3]
        blend_ratio_ub = jnp.square(params.blend_ratio_ub)
        blend_ratio_lb = jnp.square(params.blend_ratio_lb)
        blend_ratio = (v_x - blend_ratio_ub) / (blend_ratio_lb + 1E-6)
        blend_ratio = blend_ratio.squeeze()
        lambda_blend = jnp.min(jnp.asarray([
            jnp.max(jnp.asarray([blend_ratio, 0])), 1])
        )
        dx_kin_full = self._compute_dx_kin(x, u, params)
        dx_dyn = self._ode_dyn(x=x, u=u, params=params)
        dx_blend = lambda_blend * dx_dyn + (1 - lambda_blend) * dx_kin_full
        dx = (1 - use_kin) * dx_blend + use_kin * dx_kin_full
        return dx

    def _ode(self, x, u, params: CarParams):
        """
        Using kinematic model with blending: https://arxiv.org/pdf/1905.05150.pdf
        Code based on: https://github.com/alexliniger/gym-racecar/

        Inputs:
        ------
        x: jnp.ndarray,
            shape = (6, ) -> [x, y, theta, vel_r, vel_t, angular_velocity_z]
        u: jnp.ndarray,
            shape = (2, ) -> [steering_angle, throttle]
        params: CarParams,

        Output:
        -------
        dx: jnp.ndarray, derivative of x
        """
        delta, d = u[0], u[1]
        delta = jnp.clip(delta, a_min=-1, a_max=1) * params.steering_limit
        d = jnp.clip(d, a_min=-1., a_max=1)  # throttle
        u = u.at[0].set(delta)
        u = u.at[1].set(d)
        dx = self._compute_dx(x, u, params)
        return dx


class RCCarEnvReward:
    _angle_idx: int = 2
    dim_action: Tuple[int] = (2,)

    def __init__(self, goal: jnp.array, encode_angle: bool = False, ctrl_cost_weight: float = 0.005,
                 bound: float = 0.1, margin_factor: float = 10.0):
        self.goal = goal
        self.ctrl_cost_weight = ctrl_cost_weight
        self.encode_angle = encode_angle
        # Margin 20 seems to work even better (maybe try at some point)
        self.tolerance_reward = ToleranceReward(bounds=(0.0, bound), margin=margin_factor * bound,
                                                value_at_margin=0.1, sigmoid='long_tail')

    def forward(self, obs: jnp.array, action: jnp.array, next_obs: jnp.array):
        """ Computes the reward for the given transition """
        reward_ctrl = self.action_reward(action)
        reward_state = self.state_reward(obs, next_obs)
        reward = reward_state + self.ctrl_cost_weight * reward_ctrl
        return reward

    @staticmethod
    def action_reward(action: jnp.array) -> jnp.array:
        """ Computes the reward/penalty for the given action """
        return - (action ** 2).sum(-1)

    def state_reward(self, obs: jnp.array, next_obs: jnp.array) -> jnp.array:
        """ Computes the reward for the given observations """
        if self.encode_angle:
            next_obs = decode_angles(next_obs, angle_idx=self._angle_idx)
        pos_diff = next_obs[..., :2] - self.goal[:2]
        theta_diff = next_obs[..., 2] - self.goal[2]
        pos_dist = jnp.sqrt(jnp.sum(jnp.square(pos_diff), axis=-1))
        theta_dist = jnp.abs(((theta_diff + jnp.pi) % (2 * jnp.pi)) - jnp.pi)
        total_dist = jnp.sqrt(pos_dist ** 2 + theta_dist ** 2)
        reward = self.tolerance_reward(total_dist)
        return reward

    def __call__(self, *args, **kwargs):
        self.forward(*args, **kwargs)


class RCCar(Env):
    max_steps: int = 200
    base_dt: float = 1 / 30.
    dim_action: Tuple[int] = (2,)
    _goal: jnp.array = jnp.array([0.0, 0.0, 0.0])
    _init_pose: jnp.array = jnp.array([1.42, -1.04, jnp.pi])
    _angle_idx: int = 2
    _obs_noise_stds: jnp.array = OBS_NOISE_STD_SIM_CAR

    def __init__(self,
                 ctrl_cost_weight: float = 0.005,
                 encode_angle: bool = True,
                 use_obs_noise: bool = False,
                 use_tire_model: bool = True,
                 action_delay: float = 0.0,
                 car_model_params: dict = None,
                 margin_factor: float = 10.0,
                 max_throttle: float = 1.0,
                 car_id: int = 2,
                 ctrl_diff_weight: float = 0.0,
                 seed: int = 230492394,
                 max_steps: int = 200,
                 domain_randomization: bool = True,
                 sample_init_pos: bool = True,
                 dt: float | None = None,
                 dt_divisor: float = 1.0):
        """
        Race car simulator environment

        Args:
            ctrl_cost_weight: weight of the control penalty
            encode_angle: whether to encode the angle as cos(theta), sin(theta)
            use_obs_noise: whether to use observation noise
            use_tire_model: whether to use the (high-fidelity) tire model, if False just uses a kinematic bicycle model
            action_delay: whether to delay the action by a certain amount of time (in seconds)
            car_model_params: dictionary of car model parameters that overwrite the default values
            seed: random number generator seed
        """
        if dt is None:
            self._dt = self.base_dt * dt_divisor
        else:
            self._dt = dt * dt_divisor
        self.dim_state: Tuple[int] = (7,) if encode_angle else (6,)
        self.encode_angle: bool = encode_angle
        self._rds_key = jax.random.PRNGKey(seed)
        self.max_throttle = jnp.clip(max_throttle, 0.0, 1.0)
        self.max_steps = max_steps

        # set car id ant corresponding parameters
        assert car_id in [1, 2]
        self.car_id = car_id
        self._set_car_params()

        # initialize dynamics ant observation noise models
        self._dynamics_model = RaceCar(dt=self._dt, encode_angle=False) # dynamics system

        self.use_tire_model = use_tire_model
        if use_tire_model:
            self._default_car_model_params = self._default_car_model_params_blend
        else:
            self._default_car_model_params = self._default_car_model_params_bicycle

        _bounds_car_model_params = self._bounds_car_model_params_blend if use_tire_model \
            else self._bounds_car_model_params_bicycle
        self._lower_bound_params = CarParams(**{k: jnp.array(v[0]) for k, v in _bounds_car_model_params.items()})
        self._upper_bound_params = CarParams(**{k: jnp.array(v[1]) for k, v in _bounds_car_model_params.items()})

        if car_model_params is None:
            _car_model_params = self._default_car_model_params
        else:
            _car_model_params = self._default_car_model_params
            _car_model_params.update(car_model_params)
        self._dynamics_params = CarParams(**_car_model_params)
        self._next_step_fn = jax.jit(partial(self._dynamics_model.next_step, params=self._dynamics_params))

        self.use_obs_noise = use_obs_noise

        self.domain_randomization = domain_randomization
        self.sample_init_pos = sample_init_pos

        # initialize reward model
        self._reward_model = RCCarEnvReward(goal=self._goal,
                                            ctrl_cost_weight=ctrl_cost_weight,
                                            encode_angle=self.encode_angle,
                                            margin_factor=margin_factor)

        # set up action delay
        assert action_delay >= 0.0, "Action delay must be non-negative"
        self.action_delay = action_delay
        if abs(action_delay % self._dt) < 1e-8:
            self._act_delay_interpolation_weights = jnp.array([1.0, 0.0])
        else:
            # if action delay is not a multiple of dt, compute weights to interpolate
            # between temporally closest actions
            weight_first = (action_delay % self._dt) / self._dt
            self._act_delay_interpolation_weights = jnp.array([weight_first, 1.0 - weight_first])
        action_delay_buffer_size = int(jnp.ceil(action_delay / self._dt)) + 1
        self._action_buffer = jnp.zeros((action_delay_buffer_size, self.dim_action[0]))

        # initialize time ant state
        self._time: int = 0
        self._state: jnp.array = jnp.zeros(self.dim_state)
        self.ctrl_diff_weight = ctrl_diff_weight

    def _set_car_params(self):
        from envs.rccar.sim.rccar_config import (DEFAULT_PARAMS_BICYCLE_CAR1, DEFAULT_PARAMS_BLEND_CAR1,
                                           DEFAULT_PARAMS_BICYCLE_CAR2, DEFAULT_PARAMS_BLEND_CAR2,
                                           BOUNDS_PARAMS_BICYCLE_CAR1, BOUNDS_PARAMS_BLEND_CAR1,
                                           BOUNDS_PARAMS_BICYCLE_CAR2, BOUNDS_PARAMS_BLEND_CAR2)
        if self.car_id == 1:
            self._default_car_model_params_bicycle: Dict = DEFAULT_PARAMS_BICYCLE_CAR1
            self._bounds_car_model_params_bicycle: Dict = BOUNDS_PARAMS_BICYCLE_CAR1
            self._default_car_model_params_blend: Dict = DEFAULT_PARAMS_BLEND_CAR1
            self._bounds_car_model_params_blend: Dict = BOUNDS_PARAMS_BLEND_CAR1
        elif self.car_id == 2:
            self._default_car_model_params_bicycle: Dict = DEFAULT_PARAMS_BICYCLE_CAR2
            self._bounds_car_model_params_bicycle: Dict = BOUNDS_PARAMS_BICYCLE_CAR2
            self._default_car_model_params_blend: Dict = DEFAULT_PARAMS_BLEND_CAR2
            self._bounds_car_model_params_blend: Dict = BOUNDS_PARAMS_BLEND_CAR2
        else:
            raise NotImplementedError(f'Car idx {self.car_id} not supported')

    def _state_to_obs(self, state: jnp.array, rng_key: jax.random.PRNGKey) -> jnp.array:
        """ Adds observation noise to the state """
        assert state.shape[-1] == 6
        # add observation noise
        if self.use_obs_noise:
            obs = state + self._obs_noise_stds * jax.random.normal(rng_key, shape=self._state.shape)
        else:
            obs = state

        # encode angle to sin(theta) ant cos(theta) if desired
        if self.encode_angle:
            obs = encode_angles(obs, self._angle_idx)
        assert (obs.shape[-1] == 7 and self.encode_angle) or (obs.shape[-1] == 6 and not self.encode_angle)
        return obs

    def sample_params(self, rng_key: jax.random.PRNGKey):
        params = self._dynamics_model.sample_params_uniform(rng_key,
                                                  lower_bound=self._lower_bound_params,
                                                  upper_bound=self._upper_bound_params)
        return params
    def reset(self,
              rng: jax.Array) -> State:
        """ Resets the environment to a random initial state close to the initial pose """
        # sample random initial state
        key_pos, key_theta, key_vel, key_obs, key_dr = jax.random.split(rng, 5)
        if self.sample_init_pos: #sampling the initial position
            x_key, y_key = jax.random.split(key_pos, 2)
            init_x = jax.random.uniform(x_key, shape=(1,), minval=1.0, maxval=3.0)
            init_y = jax.random.uniform(y_key, shape=(1,), minval=-2.3, maxval=1.5)
            init_pos = jnp.concatenate([init_x, init_y])
            init_theta = self._init_pose[2:] + jax.random.uniform(
                key_pos, shape=(1,), minval=-jnp.pi, maxval=jnp.pi
            )
        else:
            init_pos = self._init_pose[:2] + jax.random.uniform(key_pos, shape=(2,), minval=-0.10, maxval=0.10)
            init_theta = self._init_pose[2:] + \
                         jax.random.uniform(key_pos, shape=(1,), minval=-0.10 * jnp.pi, maxval=0.10 * jnp.pi)
        init_vel = jnp.zeros((3,)) + jnp.array(
            [0.005, 0.005, 0.02]
        ) * jax.random.normal(key_vel, shape=(3,))
        init_state = jnp.concatenate([init_pos, init_theta, init_vel])

        if self.domain_randomization:  # Apply domain randomization for training
            params = self.sample_params(key_dr)
        else:  # if false, keep the default parameters.
            params = CarParams(**self._default_car_model_params)

        car_pipeline_state = CarPipelineState(car_params =params)

        # Apply observation noise and encode angle
        init_state = self._state_to_obs(init_state, rng_key=key_obs)

        # Reset time and action buffer (not jittable like this)
        #self._time = 0
        #self._action_buffer = jnp.zeros_like(self._action_buffer)
        return State(pipeline_state=car_pipeline_state,
                     obs=init_state,
                     reward=jnp.array(0.0),
                     done=jnp.array(0.0), )

    def _get_delayed_action(self, action: jnp.array) -> Tuple[jnp.array, jnp.array]:
        # push action to action buffer
        last_action = self._action_buffer[-1]
        reward = - self.ctrl_diff_weight * jnp.sum((action - last_action) ** 2)
        self._action_buffer = jnp.concatenate([self._action_buffer[1:], action[None, :]], axis=0)

        # get delayed action (interpolate between two actions if the delay is not a multiple of dt)
        delayed_action = jnp.sum(self._action_buffer[:2] * self._act_delay_interpolation_weights[:, None], axis=0)
        assert delayed_action.shape == self.dim_action
        return delayed_action, reward

    def step(self,
             state: State,
             action: jax.Array) -> State:
        assert action.shape[-1:] == self.dim_action
        action = jnp.clip(action, -1.0, 1.0)
        action = action.at[0].set(self.max_throttle * action[0])
        # assert jnp.all(-1 <= action) ant jnp.all(action <= 1), "action must be in [-1, 1]"
        jitter_reward = jnp.zeros_like(action).sum(-1)
        if self.action_delay > 0.0:
            raise NotImplementedError('Action delay is not implemented yet')
            # pushes action to action buffer ant pops the oldest action
            # computes delayed action as a linear interpolation between the relevant actions in the past
            action, jitter_reward = self._get_delayed_action(action)

        obs = state.obs
        # compute next state
        if self.encode_angle:
            obs = decode_angles(obs, self._angle_idx)
        # Move forward
        assert obs.shape[-1] == 6
        obs = self._dynamics_model.next_step(obs, action, params=state.pipeline_state.car_params)
        # TODO: if we want to handle observation noise we have to pass key to pipeline_state
        obs = self._state_to_obs(obs, rng_key=jax.random.PRNGKey(0))

        # compute reward
        reward = self._reward_model.forward(obs=None, action=action, next_obs=obs) * self._dt / self.base_dt
        # check if done
        done = self._time >= self.max_steps
        next_done = 1 - (1 - done) * (1 - state.done)

        next_state = State(pipeline_state=state.pipeline_state,
                           obs=obs,
                           reward=reward,
                           done=next_done,
                           metrics=state.metrics,
                           info=state.info)
        return next_state

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value):
        self._dt = value

    @property
    def observation_size(self) -> int:
        if self.encode_angle:
            return 7
        else:
            return 6

    @property
    def action_size(self) -> int:
        # [steering, throttle]
        return 2

    def backend(self) -> str:
        return 'positional'


if __name__ == '__main__':
    env = RCCar(dt=0.001)
    state = env.reset(rng=jax.random.PRNGKey(0))
    action = jnp.array([0.0, 0.0])

    reward = env._reward_model.forward(obs=None, action=action, next_obs=state.obs) * env._dt / env.base_dt
    print(reward)

    # for _ in range(10):
    #     state = env.step(state, action)
    #     print(state)
