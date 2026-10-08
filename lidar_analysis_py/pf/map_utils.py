import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import yaml
from PIL import Image
from scipy.ndimage import distance_transform_edt

from nav_msgs.msg import OccupancyGrid


@dataclass
class OccupancyMap:
    image_path: str
    resolution: float
    origin: Tuple[float, float, float]
    occupied: np.ndarray
    free: np.ndarray
    distance_m: np.ndarray
    width: int
    height: int
    max_likelihood_dist_m: float = 2.0

    def world_to_map(self, x, y):
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)

        col = np.floor((x_arr - self.origin[0]) / self.resolution).astype(np.int32)
        row_from_bottom = np.floor((y_arr - self.origin[1]) / self.resolution).astype(np.int32)
        row = (self.height - 1) - row_from_bottom

        valid = (
            (row >= 0)
            & (row < self.height)
            & (col >= 0)
            & (col < self.width)
        )

        return row, col, valid

    def sample_free_particles(self, n_particles: int, rng: np.random.Generator):
        free_rows, free_cols = np.where(self.free)

        if len(free_rows) == 0:
            raise RuntimeError("No free cells found in the map. Check map thresholds.")

        indices = rng.choice(len(free_rows), size=n_particles, replace=True)
        rows = free_rows[indices]
        cols = free_cols[indices]

        x = self.origin[0] + (cols + 0.5) * self.resolution
        row_from_bottom = (self.height - 1) - rows
        y = self.origin[1] + (row_from_bottom + 0.5) * self.resolution
        yaw = rng.uniform(-np.pi, np.pi, size=n_particles)

        return np.column_stack((x, y, yaw)).astype(np.float64)

    def to_occupancy_grid_msg(self, stamp, frame_id: str):
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id

        msg.info.resolution = float(self.resolution)
        msg.info.width = int(self.width)
        msg.info.height = int(self.height)

        msg.info.origin.position.x = float(self.origin[0])
        msg.info.origin.position.y = float(self.origin[1])
        msg.info.origin.position.z = 0.0

        yaw = float(self.origin[2]) if len(self.origin) >= 3 else 0.0
        msg.info.origin.orientation.z = float(np.sin(yaw / 2.0))
        msg.info.origin.orientation.w = float(np.cos(yaw / 2.0))

        grid = np.full((self.height, self.width), -1, dtype=np.int8)
        grid[self.free] = 0
        grid[self.occupied] = 100

        # OccupancyGrid data starts at map cell (0,0), which corresponds to the lower-left.
        grid_bottom_first = np.flipud(grid)
        msg.data = [int(v) for v in grid_bottom_first.reshape(-1)]

        return msg


def load_occupancy_map(yaml_path: str, max_likelihood_dist_m: float = 2.0) -> OccupancyMap:
    yaml_path = os.path.expanduser(yaml_path)

    with open(yaml_path, "r") as f:
        metadata = yaml.safe_load(f)

    image_path = metadata["image"]
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(yaml_path), image_path)

    resolution = float(metadata["resolution"])
    origin = tuple(float(v) for v in metadata.get("origin", [0.0, 0.0, 0.0]))
    negate = int(metadata.get("negate", 0))
    occupied_thresh = float(metadata.get("occupied_thresh", 0.65))
    free_thresh = float(metadata.get("free_thresh", 0.25))

    image = np.asarray(Image.open(image_path).convert("L"), dtype=np.float64)

    if negate == 0:
        occ_prob = (255.0 - image) / 255.0
    else:
        occ_prob = image / 255.0

    occupied = occ_prob > occupied_thresh
    free = occ_prob < free_thresh

    # Unknown cells are treated as obstacles for the likelihood field.
    # distance_transform_edt returns distance to the nearest zero cell.
    # Since free=True and occupied/unknown=False, free cells get distance to nearest obstacle/unknown.
    distance_m = distance_transform_edt(free) * resolution
    distance_m = np.minimum(distance_m, max_likelihood_dist_m)

    return OccupancyMap(
        image_path=image_path,
        resolution=resolution,
        origin=origin,
        occupied=occupied,
        free=free,
        distance_m=distance_m,
        width=image.shape[1],
        height=image.shape[0],
        max_likelihood_dist_m=max_likelihood_dist_m,
    )
