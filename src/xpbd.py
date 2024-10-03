import jax
import jax.numpy as jnp
jnp.set_printoptions(precision=32, suppress=True)

rest_length = 1.3
mass = 4.0
cube = jnp.array([
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [-1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0]
]).astype(jnp.float32) * (rest_length / 2.0)
rest = cube.copy()
rest_com = jnp.mean(rest, axis=0)
Q = jnp.zeros((3, 3))
for i in range(8):
    r = rest[i] - rest_com
    Q += mass * jnp.outer(r, r)
inv_Q = jnp.linalg.inv(Q)
print(inv_Q)
# Rotate cube by 45 degrees around x-axis
rotation_angle = jnp.pi / 4.0  # 45 degrees in radians
rotation_matrix = jnp.array([
    [1.0, 0.0, 0.0],
    [0.0, jnp.cos(rotation_angle), -jnp.sin(rotation_angle)],
    [0.0, jnp.sin(rotation_angle), jnp.cos(rotation_angle)]
], dtype=jnp.float64)
cube = jnp.dot(cube, rotation_matrix.T)

cube = cube
cube_v = jnp.zeros_like(cube)
ori_cube = cube
print(rest_com)
ts = (1.0 / 60.0) / 20.0
@jax.jit
def step(carry, _):
    cube, cube_v = carry
    cube = cube + cube_v * ts
    com = jnp.sum(cube * mass, axis=0) / (mass * 8.0)
    com = jnp.zeros_like(com)
    rest_com = jnp.zeros_like(com)
    def compute_F(F, i):
        r = cube[i] - com
        return F + mass * jnp.outer(r, rest[i] - rest_com), None

    F, _ = jax.lax.scan(compute_F, jnp.zeros((3, 3)), jnp.arange(8))
    F = F @ inv_Q

    new_cube = (F @ rest.T).T + com
    dx = new_cube - cube
    cube_v = dx / ts
    return (new_cube, cube_v), None

(final_cube, final_cube_v), _ = jax.lax.scan(step, (cube, cube_v), None, length=20 * 60 * 60 * 10)

print(final_cube - ori_cube)

import time

# Define the number of cubes
num_cubes = 10_000_000

# Vectorize the initial conditions
cube_batch = jnp.tile(cube[None, ...], (num_cubes, 1, 1))
cube_v_batch = jnp.tile(cube_v[None, ...], (num_cubes, 1, 1))

# Vectorize the step function
vstep = jax.vmap(step, in_axes=(0, None))


# Time the execution
dt = 10000.0
for i in range(10):
    start_time = time.time()
    (final_cube_batch, final_cube_v_batch), _ = vstep((cube_batch, cube_v_batch), None)
    end_time = time.time()
    dt = min(dt, end_time - start_time)
print(f"Time taken for {num_cubes} cubes: {dt:.4f} seconds")

