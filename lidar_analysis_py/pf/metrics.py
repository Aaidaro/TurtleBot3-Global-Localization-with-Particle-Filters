import numpy as np

from .motion_model import wrap_angle


def weighted_pose_mean(particles, weights):
    x = np.sum(weights * particles[:, 0])
    y = np.sum(weights * particles[:, 1])

    sin_sum = np.sum(weights * np.sin(particles[:, 2]))
    cos_sum = np.sum(weights * np.cos(particles[:, 2]))
    yaw = np.arctan2(sin_sum, cos_sum)

    return np.array([x, y, yaw], dtype=np.float64)


def weighted_pose_spread(particles, weights, mean_pose):
    dx = particles[:, 0] - mean_pose[0]
    dy = particles[:, 1] - mean_pose[1]
    dyaw = wrap_angle(particles[:, 2] - mean_pose[2])

    std_x = np.sqrt(np.sum(weights * dx * dx))
    std_y = np.sqrt(np.sum(weights * dy * dy))
    std_yaw = np.sqrt(np.sum(weights * dyaw * dyaw))

    return np.array([std_x, std_y, std_yaw], dtype=np.float64)


def pose_error(estimate, truth):
    dx = estimate[0] - truth[0]
    dy = estimate[1] - truth[1]

    position_error = np.hypot(dx, dy)
    yaw_error = wrap_angle(estimate[2] - truth[2])

    return float(position_error), float(yaw_error)
