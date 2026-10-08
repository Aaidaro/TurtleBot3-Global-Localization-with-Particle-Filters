import math
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool


class PFExploreMotionNode(Node):
    def __init__(self):
        super().__init__("pf_explore_motion_node")

        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("converged_topic", "/pf/converged")

        self.declare_parameter("linear_speed", 0.08)
        self.declare_parameter("angular_speed", 0.55)
        self.declare_parameter("wander_angular_speed", 0.12)
        self.declare_parameter("front_clearance_m", 0.45)
        self.declare_parameter("front_sector_deg", 35.0)
        self.declare_parameter("turn_duration_sec", 1.8)
        self.declare_parameter("update_rate_hz", 10.0)
        self.declare_parameter("seed", -1)

        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.scan_topic = self.get_parameter("scan_topic").value
        self.converged_topic = self.get_parameter("converged_topic").value

        seed = int(self.get_parameter("seed").value)
        self.rng = np.random.default_rng(None if seed < 0 else seed)

        self.latest_scan = None
        self.converged = False
        self.turn_until_time = 0.0
        self.turn_direction = 1.0

        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Bool,
            self.converged_topic,
            self.converged_callback,
            10,
        )

        rate = float(self.get_parameter("update_rate_hz").value)
        self.timer = self.create_timer(1.0 / rate, self.control_loop)

        self.get_logger().info("PF exploration motion node started.")
        self.get_logger().info("Robot will move until /pf/converged is true.")

    def scan_callback(self, msg):
        self.latest_scan = msg

    def converged_callback(self, msg):
        if msg.data and not self.converged:
            self.get_logger().info("PF converged. Stopping robot.")
        self.converged = bool(msg.data)

    def stop(self):
        self.cmd_pub.publish(Twist())

    def get_front_min_range(self):
        if self.latest_scan is None:
            return float("inf")

        scan = self.latest_scan
        ranges = np.asarray(scan.ranges, dtype=np.float64)

        if len(ranges) == 0:
            return float("inf")

        angles = scan.angle_min + np.arange(len(ranges)) * scan.angle_increment
        sector = math.radians(float(self.get_parameter("front_sector_deg").value))

        mask = (
            np.isfinite(ranges)
            & (ranges >= scan.range_min)
            & (ranges <= scan.range_max)
            & (np.abs(angles) <= sector)
        )

        if not np.any(mask):
            return float("inf")

        return float(np.min(ranges[mask]))

    def control_loop(self):
        if self.converged:
            self.stop()
            return

        now = self.get_clock().now().nanoseconds * 1e-9

        linear_speed = float(self.get_parameter("linear_speed").value)
        angular_speed = float(self.get_parameter("angular_speed").value)
        wander_angular_speed = float(self.get_parameter("wander_angular_speed").value)
        clearance = float(self.get_parameter("front_clearance_m").value)
        turn_duration = float(self.get_parameter("turn_duration_sec").value)

        front_min = self.get_front_min_range()

        cmd = Twist()

        if now < self.turn_until_time:
            cmd.angular.z = self.turn_direction * angular_speed
            self.cmd_pub.publish(cmd)
            return

        if front_min < clearance:
            self.turn_direction = float(self.rng.choice([-1.0, 1.0]))
            self.turn_until_time = now + turn_duration

            cmd.angular.z = self.turn_direction * angular_speed
            self.cmd_pub.publish(cmd)
            return

        cmd.linear.x = linear_speed
        cmd.angular.z = float(self.rng.uniform(-wander_angular_speed, wander_angular_speed))
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = PFExploreMotionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop()
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
