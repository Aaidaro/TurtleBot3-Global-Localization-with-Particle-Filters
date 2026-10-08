import math
import os
import time
from pathlib import Path

import numpy as np

import rclpy
from rclpy.node import Node

from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from geometry_msgs.msg import Pose

from .pf.map_utils import load_occupancy_map
from .pf.motion_model import wrap_angle


def pose_from_xyyaw(x, y, yaw, z=0.01):
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = float(z)
    pose.orientation.z = float(math.sin(yaw / 2.0))
    pose.orientation.w = float(math.cos(yaw / 2.0))
    return pose


def map_pose_to_truth_pose(x_map, y_map, yaw_map, x_offset, y_offset, yaw_offset):
    dx = x_map - x_offset
    dy = y_map - y_offset

    c = math.cos(-yaw_offset)
    s = math.sin(-yaw_offset)

    x_truth = c * dx - s * dy
    y_truth = s * dx + c * dy
    yaw_truth = wrap_angle(yaw_map - yaw_offset)

    return x_truth, y_truth, yaw_truth


def find_turtlebot3_sdf(model_name: str):
    tb3_gazebo_share = get_package_share_directory("turtlebot3_gazebo")

    candidates = [
        Path(tb3_gazebo_share) / "models" / model_name / "model.sdf",
    ]

    # CHATGPT handled this:
    if not model_name.startswith("turtlebot3_"):
        candidates.append(
            Path(tb3_gazebo_share) / "models" / f"turtlebot3_{model_name}" / "model.sdf"
        )

    for path in candidates:
        if path.exists():
            return str(path)

    raise FileNotFoundError(
        "Could not find TurtleBot3 model.sdf. Tried:\n"
        + "\n".join(str(p) for p in candidates)
    )


class RandomizeRobotPose(Node):
    def __init__(self):
        super().__init__("randomize_robot_pose")

        default_map = str(Path.home() / "tb3_projects_ws/maps/explore_house.yaml")

        self.declare_parameter("map_yaml_path", default_map)
        self.declare_parameter("model_name", "turtlebot3_burger")
        self.declare_parameter("model_sdf_path", "")

        self.declare_parameter("delete_service", "/delete_entity")
        self.declare_parameter("spawn_service", "/spawn_entity")
        self.declare_parameter("reference_frame", "world")

        # Your calibrated truth transform.
        self.declare_parameter("truth_x_offset", 2.229376)
        self.declare_parameter("truth_y_offset", 0.415668)
        self.declare_parameter("truth_yaw_offset", -0.016549)

        self.declare_parameter("min_clearance_m", 0.35)
        self.declare_parameter("robot_z", 0.01)
        self.declare_parameter("seed", -1)

        self.map_yaml_path = self.get_parameter("map_yaml_path").value
        self.model_name = self.get_parameter("model_name").value
        self.model_sdf_path = self.get_parameter("model_sdf_path").value

        self.delete_service = self.get_parameter("delete_service").value
        self.spawn_service = self.get_parameter("spawn_service").value
        self.reference_frame = self.get_parameter("reference_frame").value

        self.truth_x_offset = float(self.get_parameter("truth_x_offset").value)
        self.truth_y_offset = float(self.get_parameter("truth_y_offset").value)
        self.truth_yaw_offset = float(self.get_parameter("truth_yaw_offset").value)

        self.min_clearance_m = float(self.get_parameter("min_clearance_m").value)
        self.robot_z = float(self.get_parameter("robot_z").value)

        seed = int(self.get_parameter("seed").value)
        self.rng = np.random.default_rng(None if seed < 0 else seed)

        self.occ_map = load_occupancy_map(self.map_yaml_path)

        if not self.model_sdf_path:
            self.model_sdf_path = find_turtlebot3_sdf(self.model_name)

        with open(self.model_sdf_path, "r") as f:
            self.model_xml = f.read()

        self.delete_client = self.create_client(DeleteEntity, self.delete_service)
        self.spawn_client = self.create_client(SpawnEntity, self.spawn_service)

        self.timer = self.create_timer(0.2, self.run_once)
        self.started = False

        self.get_logger().info(f"Map: {self.map_yaml_path}")
        self.get_logger().info(f"Model name: {self.model_name}")
        self.get_logger().info(f"SDF path: {self.model_sdf_path}")
        self.get_logger().info(f"Delete service: {self.delete_service}")
        self.get_logger().info(f"Spawn service: {self.spawn_service}")

    def sample_random_map_pose(self):
        valid = self.occ_map.free & (self.occ_map.distance_m >= self.min_clearance_m)

        rows, cols = np.where(valid)
        if len(rows) == 0:
            self.get_logger().warn("No cells satisfy min_clearance_m. Falling back to all free cells.")
            rows, cols = np.where(self.occ_map.free)

        if len(rows) == 0:
            raise RuntimeError("No free cells found in map.")

        idx = self.rng.integers(0, len(rows))
        row = int(rows[idx])
        col = int(cols[idx])

        x_map = self.occ_map.origin[0] + (col + 0.5) * self.occ_map.resolution
        row_from_bottom = (self.occ_map.height - 1) - row
        y_map = self.occ_map.origin[1] + (row_from_bottom + 0.5) * self.occ_map.resolution
        yaw_map = self.rng.uniform(-math.pi, math.pi)

        return x_map, y_map, yaw_map

    def call_delete(self):
        req = DeleteEntity.Request()
        req.name = self.model_name

        future = self.delete_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if not future.done():
            self.get_logger().warn("DeleteEntity timed out. Continuing to spawn anyway.")
            return

        res = future.result()
        if res is None:
            self.get_logger().warn("DeleteEntity returned no response. Continuing.")
            return

        if res.success:
            self.get_logger().info(f"Deleted existing entity: {self.model_name}")
        else:
            self.get_logger().warn(f"DeleteEntity reported failure: {res.status_message}")

    def call_spawn(self, x_gazebo, y_gazebo, yaw_gazebo):
        req = SpawnEntity.Request()
        req.name = self.model_name
        req.xml = self.model_xml
        req.robot_namespace = ""
        req.initial_pose = pose_from_xyyaw(x_gazebo, y_gazebo, yaw_gazebo, self.robot_z)
        req.reference_frame = self.reference_frame

        future = self.spawn_client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if not future.done():
            self.get_logger().error("SpawnEntity timed out.")
            return False

        res = future.result()
        if res is None:
            self.get_logger().error("SpawnEntity returned no response.")
            return False

        if res.success:
            self.get_logger().info(f"Spawned {self.model_name} at randomized pose.")
            return True

        self.get_logger().error(f"SpawnEntity failed: {res.status_message}")
        return False

    def run_once(self):
        if self.started:
            return

        if not self.delete_client.wait_for_service(timeout_sec=0.1):
            return

        if not self.spawn_client.wait_for_service(timeout_sec=0.1):
            return

        self.started = True

        x_map, y_map, yaw_map = self.sample_random_map_pose()

        x_gazebo, y_gazebo, yaw_gazebo = map_pose_to_truth_pose(
            x_map,
            y_map,
            yaw_map,
            self.truth_x_offset,
            self.truth_y_offset,
            self.truth_yaw_offset,
        )

        self.get_logger().info("Random sampled pose:")
        self.get_logger().info(f"  map:    x={x_map:.3f}, y={y_map:.3f}, yaw={yaw_map:.3f}")
        self.get_logger().info(f"  gazebo: x={x_gazebo:.3f}, y={y_gazebo:.3f}, yaw={yaw_gazebo:.3f}")

        self.call_delete()
        time.sleep(0.7)
        ok = self.call_spawn(x_gazebo, y_gazebo, yaw_gazebo)

        if ok:
            self.get_logger().info("Random initialization complete. Now start PF.")
        else:
            self.get_logger().error("Random initialization failed.")

        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = RandomizeRobotPose()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
