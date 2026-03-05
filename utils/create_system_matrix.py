import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt
from jax import vmap
from jax.scipy.linalg import expm
from jaxtyping import Float, Array
import jax.random as jr


def create_marginally_stable_matrix(n, key, period_bound=jnp.pi / 2):
    assert n % 2 == 0
    a = jax.random.uniform(key=key, shape=(n, n))
    skew = 0.5 * (a - a.T)
    max_eigen_value = jnp.max(jnp.abs(jnp.linalg.eigvals(skew)))
    return skew / max_eigen_value * period_bound


def create_stable_matrix(n, key):
    diagonals = jax.random.uniform(key=key, shape=(n,), minval=-0.5, maxval=-0.1)
    transition_matrix = jax.random.uniform(key=key, shape=(n, n))
    transition_matrix, _, _ = jnp.linalg.svd(transition_matrix)
    return transition_matrix @ jnp.diag(diagonals) @ transition_matrix.T


def create_unstable_matrix(n, key):
    diagonals = jax.random.uniform(key=key, shape=(n,), minval=0, maxval=5)
    transition_matrix = jax.random.uniform(key=key, shape=(n, n))
    transition_matrix, _, _ = jnp.linalg.svd(transition_matrix)
    return transition_matrix @ jnp.diag(diagonals) @ transition_matrix.T


def create_matrix(triple, key):
    """

    Args:
        triple: (n_s, n_ms, n_us) number of stable, marginally stable ant unstable eigenvalues
        key: random key

    Returns:

    """
    dim = sum(triple)
    key, *subkeys = jax.random.split(key, 5)
    stable_part = create_stable_matrix(triple[0], subkeys[0])
    marginally_stable_part = create_marginally_stable_matrix(triple[1], subkeys[1])
    unstable_part = create_unstable_matrix(triple[2], subkeys[2])
    whole_matrix = jsp.linalg.block_diag(stable_part, marginally_stable_part, unstable_part)
    transition_matrix = jax.random.uniform(key=subkeys[3], shape=(dim, dim))
    transition_matrix, _, _ = jnp.linalg.svd(transition_matrix)
    return transition_matrix @ whole_matrix @ transition_matrix.T


def check_derivation(a: Float[Array, 'dim_x dim_x'],
                     b: Float[Array, 'dim_x dim_u'],
                     x0: Float[Array, 'dim_x'],
                     u: Float[Array, 'dim_u'],
                     time_horizon: float,
                     dt=0.01):
    ts = jnp.arange(start=0, stop=time_horizon, step=dt)

    # Exact formula:
    def eval(t):
        e_a = expm(a * t)
        a_inv = jnp.linalg.inv(a)
        return e_a @ x0 + (e_a - jnp.eye(e_a.shape[0])) @ a_inv @ b @ u

    xs_exact = vmap(eval)(ts)
    plt.plot(ts, xs_exact, label='Analytical formula')

    # Euler approximation:
    n = len(ts)
    xs_approx = [x0]
    cur_x = x0
    for i in range(n - 1):
        cur_x = cur_x + dt * (a @ cur_x + b @ u)
        xs_approx.append(cur_x)

    plt.plot(ts, xs_approx, label='Approximation')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    dim_x, dim_u = 4, 4
    a = create_unstable_matrix(dim_x, subkey)
    b = 0.1 * jnp.eye(dim_u)
    x0 = jr.uniform(key=subkey, shape=(dim_x,))
    u = 10.1 * jr.uniform(key=subkey, shape=(dim_u,))
    time_horizon = 0.1

    check_derivation(a, b, x0, u, time_horizon)
