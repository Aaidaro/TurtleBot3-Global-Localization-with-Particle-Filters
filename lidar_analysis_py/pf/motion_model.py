import numpy as np


def wrap_angle(angle):
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def odometry_delta(prev_pose, curr_pose):
    x0, y0, th0 = prev_pose
    x1, y1, th1 = curr_pose

    dx = x1 - x0
    dy = y1 - y0

    delta_trans = np.hypot(dx, dy)

    if delta_trans < 1e-9:
        delta_rot1 = 0.0
    else:
        delta_rot1 = wrap_angle(np.arctan2(dy, dx) - th0)

    delta_rot2 = wrap_angle(th1 - th0 - delta_rot1)

    return float(delta_rot1), float(delta_trans), float(delta_rot2)


def apply_odometry_motion(particles, delta, alphas, rng):
    delta_rot1, delta_trans, delta_rot2 = delta
    a1, a2, a3, a4 = alphas

    var_rot1 = a1 * delta_rot1**2 + a2 * delta_trans**2
    var_trans = a3 * delta_trans**2 + a4 * (delta_rot1**2 + delta_rot2**2)
    var_rot2 = a1 * delta_rot2**2 + a2 * delta_trans**2

    std_rot1 = np.sqrt(max(var_rot1, 1e-12))
    std_trans = np.sqrt(max(var_trans, 1e-12))
    std_rot2 = np.sqrt(max(var_rot2, 1e-12))

    n = particles.shape[0]

    noisy_rot1 = delta_rot1 + rng.normal(0.0, std_rot1, size=n)
    noisy_trans = delta_trans + rng.normal(0.0, std_trans, size=n)
    noisy_rot2 = delta_rot2 + rng.normal(0.0, std_rot2, size=n)

    theta = particles[:, 2]

    particles[:, 0] += noisy_trans * np.cos(theta + noisy_rot1)
    particles[:, 1] += noisy_trans * np.sin(theta + noisy_rot1)
    particles[:, 2] = wrap_angle(theta + noisy_rot1 + noisy_rot2)

    return particles
