import math
import numpy as np

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def rotate_xy(x, y, yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    return c * x - s * y, s * x + c * y


class TruthTransformCalibrator(Node):
    def __init__(self):
        super().__init__("truth_transform_calibrator")

        self.declare_parameter("truth_topic", "/ground_truth_odom")
        self.declare_parameter("estimate_topic", "/pf/estimated_pose")
        self.declare_parameter("sample_count", 80)

        self.truth_topic = self.get_parameter("truth_topic").value
        self.estimate_topic = self.get_parameter("estimate_topic").value
        self.sample_count = int(self.get_parameter("sample_count").value)

        self.truth = None
        self.estimate = None
        self.truth_samples = []
        self.estimate_samples = []

        self.create_subscription(Odometry, self.truth_topic, self.truth_cb, 10)
        self.create_subscription(PoseStamped, self.estimate_topic, self.estimate_cb, 10)

        self.timer = self.create_timer(0.2, self.tick)

        self.get_logger().info(f"Listening to truth: {self.truth_topic}")
        self.get_logger().info(f"Listening to estimate: {self.estimate_topic}")
        self.get_logger().info("Keep robot still while this samples.")

    def truth_cb(self, msg):
        p = msg.pose.pose.position
        yaw = yaw_from_quat(msg.pose.pose.orientation)
        self.truth = np.array([p.x, p.y, yaw], dtype=float)

    def estimate_cb(self, msg):
        p = msg.pose.position
        yaw = yaw_from_quat(msg.pose.orientation)
        self.estimate = np.array([p.x, p.y, yaw], dtype=float)

    def tick(self):
        if self.truth is None or self.estimate is None:
            return

        self.truth_samples.append(self.truth.copy())
        self.estimate_samples.append(self.estimate.copy())

        if len(self.truth_samples) % 10 == 0:
            self.get_logger().info(f"Collected {len(self.truth_samples)}/{self.sample_count} samples")

        if len(self.truth_samples) < self.sample_count:
            return

        truth_arr = np.asarray(self.truth_samples)
        est_arr = np.asarray(self.estimate_samples)

        yaw_offsets = np.array([
            wrap_angle(e[2] - t[2])
            for t, e in zip(truth_arr, est_arr)
        ])

        truth_yaw_offset = float(np.median(yaw_offsets))

        x_offsets = []
        y_offsets = []

        for t, e in zip(truth_arr, est_arr):
            rx, ry = rotate_xy(t[0], t[1], truth_yaw_offset)
            x_offsets.append(e[0] - rx)
            y_offsets.append(e[1] - ry)

        truth_x_offset = float(np.median(x_offsets))
        truth_y_offset = float(np.median(y_offsets))

        self.get_logger().info("")
        self.get_logger().info("=== Corrected suggested transform parameters ===")
        self.get_logger().info(f"truth_x_offset := {truth_x_offset:.6f}")
        self.get_logger().info(f"truth_y_offset := {truth_y_offset:.6f}")
        self.get_logger().info(f"truth_yaw_offset := {truth_yaw_offset:.6f}")
        self.get_logger().info("")
        self.get_logger().info("Run PF with:")
        self.get_logger().info(
            f"-p truth_x_offset:={truth_x_offset:.6f} "
            f"-p truth_y_offset:={truth_y_offset:.6f} "
            f"-p truth_yaw_offset:={truth_yaw_offset:.6f}"
        )
        self.get_logger().info("")
        self.get_logger().info("Stability check:")
        self.get_logger().info(
            f"std_x={np.std(x_offsets):.4f}, "
            f"std_y={np.std(y_offsets):.4f}, "
            f"std_yaw={np.std(yaw_offsets):.4f}"
        )

        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = TruthTransformCalibrator()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
