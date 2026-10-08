import numpy as np


def laser_log_likelihood(
    particles,
    scan_msg,
    occ_map,
    beam_step=8,
    max_beams=80,
    sigma_hit=0.15,
    z_hit=0.95,
    z_rand=0.05,
    max_range=3.5,
):
    ranges = np.asarray(scan_msg.ranges, dtype=np.float64)

    if len(ranges) == 0:
        return np.zeros(particles.shape[0], dtype=np.float64)

    all_indices = np.arange(0, len(ranges), max(1, int(beam_step)))
    selected_ranges = ranges[all_indices]

    scan_max_range = min(float(scan_msg.range_max), float(max_range))
    valid = (
        np.isfinite(selected_ranges)
        & (selected_ranges >= float(scan_msg.range_min))
        & (selected_ranges <= scan_max_range)
    )

    beam_indices = all_indices[valid]

    if len(beam_indices) == 0:
        return np.zeros(particles.shape[0], dtype=np.float64)

    if len(beam_indices) > max_beams:
        keep = np.linspace(0, len(beam_indices) - 1, max_beams).astype(np.int32)
        beam_indices = beam_indices[keep]

    beam_ranges = ranges[beam_indices]
    beam_angles = float(scan_msg.angle_min) + beam_indices * float(scan_msg.angle_increment)

    log_w = np.zeros(particles.shape[0], dtype=np.float64)

    px = particles[:, 0]
    py = particles[:, 1]
    pth = particles[:, 2]

    sigma2 = sigma_hit * sigma_hit
    uniform_prob = 1.0 / max(scan_max_range, 1e-6)

    for r, a in zip(beam_ranges, beam_angles):
        global_angle = pth + a

        end_x = px + r * np.cos(global_angle)
        end_y = py + r * np.sin(global_angle)

        rows, cols, valid_cells = occ_map.world_to_map(end_x, end_y)

        d = np.full(particles.shape[0], occ_map.max_likelihood_dist_m, dtype=np.float64)
        d[valid_cells] = occ_map.distance_m[rows[valid_cells], cols[valid_cells]]

        p_hit = np.exp(-0.5 * (d * d) / sigma2)
        p = z_hit * p_hit + z_rand * uniform_prob

        log_w += np.log(p + 1e-12)

    # Keep the number of beams from making weights overly sharp.
    return log_w / float(len(beam_indices))
