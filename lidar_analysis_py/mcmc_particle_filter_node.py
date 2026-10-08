import csv
import os
import time
from datetime import datetime
from pathlib import Path as FsPath

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy, qos_profile_sensor_data

from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped, PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Odometry, OccupancyGrid, Path as NavPath
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float64
from visualization_msgs.msg import Marker, MarkerArray

from .pf.map_utils import load_occupancy_map
from .pf.motion_model import odometry_delta, apply_odometry_motion, wrap_angle
from .pf.observation_model import laser_log_likelihood
from .pf.resampling import normalize_log_weights, effective_sample_size, systematic_resample
from .pf.metrics import weighted_pose_mean, weighted_pose_spread, pose_error


def quaternion_to_yaw(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return np.arctan2(siny_cosp, cosy_cosp)


def odom_msg_to_pose(msg: Odometry):
    p = msg.pose.pose.position
    q = msg.pose.pose.orientation
    return np.array([p.x, p.y, quaternion_to_yaw(q)], dtype=np.float64)


def transform_truth_pose_to_map(raw_pose, x_offset, y_offset, yaw_offset):
    """Convert Gazebo/world truth pose into the PF map frame.

    x_map = R(yaw_offset) * x_truth + translation
    yaw_map = yaw_truth + yaw_offset
    """
    c = np.cos(yaw_offset)
    ss = np.sin(yaw_offset)

    x = c * raw_pose[0] - ss * raw_pose[1] + x_offset
    y = ss * raw_pose[0] + c * raw_pose[1] + y_offset
    yaw = wrap_angle(raw_pose[2] + yaw_offset)

    return np.array([x, y, yaw], dtype=np.float64)


def pose_from_xyyaw(x, y, yaw):
    pose = Pose()
    pose.position.x = float(x)
    pose.position.y = float(y)
    pose.position.z = 0.0
    pose.orientation.z = float(np.sin(yaw / 2.0))
    pose.orientation.w = float(np.cos(yaw / 2.0))
    return pose


class MCMCParticleFilterNode(Node):
    def __init__(self):
        super().__init__("mcmc_particle_filter_localization")

        default_map = str(FsPath.home() / "tb3_projects_ws/maps/explore_house.yaml")
        default_log_dir = str(FsPath.home() / "tb3_projects_ws/log/pf_mcmc_runs")

        self.declare_parameter("map_yaml_path", default_map)
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")

        self.declare_parameter("use_odom_as_truth", True)
        self.declare_parameter("true_odom_topic", "/odom")

        # Transform raw ground-truth odometry into map coordinates.
        # Useful when Gazebo publishes truth in world frame but PF estimates in map frame.
        self.declare_parameter("truth_x_offset", 0.0)
        self.declare_parameter("truth_y_offset", 0.0)
        self.declare_parameter("truth_yaw_offset", 0.0)
        self.declare_parameter("enable_truth_calibration_from_initialpose", True)
        self.declare_parameter("truth_calibration_topic", "/initialpose")

        self.declare_parameter("n_particles", 1000)
        self.declare_parameter("seed", -1)

        self.declare_parameter("alpha1", 0.02)
        self.declare_parameter("alpha2", 0.02)
        self.declare_parameter("alpha3", 0.04)
        self.declare_parameter("alpha4", 0.02)

        self.declare_parameter("beam_step", 8)
        self.declare_parameter("max_beams", 80)
        self.declare_parameter("sigma_hit", 0.15)
        self.declare_parameter("z_hit", 0.95)
        self.declare_parameter("z_rand", 0.05)
        self.declare_parameter("max_range", 3.5)
        self.declare_parameter("max_likelihood_dist_m", 2.0)

        self.declare_parameter("resample_neff_fraction", 0.5)
        self.declare_parameter("roughening_xy_std", 0.003)
        self.declare_parameter("roughening_yaw_std", 0.003)

        # MCMC Metropolis-Hastings move after resampling.
        self.declare_parameter("mcmc_enabled", True)
        self.declare_parameter("mcmc_iterations", 1)
        self.declare_parameter("mcmc_proposal_xy_std", 0.04)
        self.declare_parameter("mcmc_proposal_yaw_std", 0.08)
        self.declare_parameter("mcmc_likelihood_scale", 10.0)
        self.declare_parameter("mcmc_min_clearance_m", 0.05)

        self.declare_parameter("converged_std_xy", 0.08)
        self.declare_parameter("converged_std_yaw", 0.10)
        self.declare_parameter("converged_required_updates", 20)

        # Prevent false convergence during simulation/debugging.
        self.declare_parameter("min_scan_updates_before_convergence", 60)
        self.declare_parameter("debug_require_truth_error_for_convergence", True)
        self.declare_parameter("converged_position_error", 0.35)
        self.declare_parameter("converged_yaw_error", 0.35)

        # Kidnapped robot detection. Uses scan likelihood / odometry jump,
        # not ground truth. Ground truth is only for evaluation.
        self.declare_parameter("kidnap_detection_enabled", True)
        self.declare_parameter("kidnap_required_bad_scans", 5)
        self.declare_parameter("kidnap_log_likelihood_drop", 1.50)
        self.declare_parameter("kidnap_absolute_log_likelihood", -3.50)
        self.declare_parameter("kidnap_baseline_alpha", 0.05)
        self.declare_parameter("kidnap_odom_jump_m", 0.75)
        self.declare_parameter("kidnap_odom_jump_rad", 1.00)
        self.declare_parameter("kidnap_event_latch_scans", 25)

        self.declare_parameter("max_path_length", 5000)
        self.declare_parameter("path_publish_min_distance", 0.02)

        self.declare_parameter("log_dir", default_log_dir)

        self.map_yaml_path = self.get_parameter("map_yaml_path").value
        self.map_frame = self.get_parameter("map_frame").value
        self.odom_topic = self.get_parameter("odom_topic").value
        self.scan_topic = self.get_parameter("scan_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value

        self.use_odom_as_truth = bool(self.get_parameter("use_odom_as_truth").value)
        self.true_odom_topic = self.get_parameter("true_odom_topic").value

        self.truth_x_offset = float(self.get_parameter("truth_x_offset").value)
        self.truth_y_offset = float(self.get_parameter("truth_y_offset").value)
        self.truth_yaw_offset = float(self.get_parameter("truth_yaw_offset").value)
        self.enable_truth_calibration_from_initialpose = bool(
            self.get_parameter("enable_truth_calibration_from_initialpose").value
        )
        self.truth_calibration_topic = self.get_parameter("truth_calibration_topic").value

        self.n_particles = int(self.get_parameter("n_particles").value)
        seed = int(self.get_parameter("seed").value)
        self.rng = np.random.default_rng(None if seed < 0 else seed)

        self.alphas = np.array(
            [
                float(self.get_parameter("alpha1").value),
                float(self.get_parameter("alpha2").value),
                float(self.get_parameter("alpha3").value),
                float(self.get_parameter("alpha4").value),
            ],
            dtype=np.float64,
        )

        max_likelihood_dist_m = float(self.get_parameter("max_likelihood_dist_m").value)
        self.map = load_occupancy_map(
            self.map_yaml_path,
            max_likelihood_dist_m=max_likelihood_dist_m,
        )

        self.particles = self.map.sample_free_particles(self.n_particles, self.rng)
        self.weights = np.ones(self.n_particles, dtype=np.float64) / float(self.n_particles)

        self.prev_odom_pose = None
        self.raw_true_pose = None
        self.true_pose = None
        self.latest_estimate = None
        self.latest_spread = None
        self.latest_neff = float(self.n_particles)
        self.latest_processing_time_ms = 0.0
        self.latest_mcmc_acceptance_rate = 0.0

        self.converged_counter = 0
        self.converged = False
        self.scan_update_count = 0
        self.latest_convergence_progress = 0.0
        self.latest_scan_match_score = float('nan')

        self.scan_match_baseline = None
        self.kidnap_bad_scan_count = 0
        self.kidnap_events = 0
        self.kidnap_event_countdown = 0

        self.estimated_path = NavPath()
        self.true_path = NavPath()
        self.estimated_path.header.frame_id = self.map_frame
        self.true_path.header.frame_id = self.map_frame
        self.last_estimated_path_pose = None
        self.last_true_path_pose = None

        map_qos = QoSProfile(depth=1)
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        map_qos.reliability = ReliabilityPolicy.RELIABLE

        self.map_pub = self.create_publisher(OccupancyGrid, "/map", map_qos)
        self.particle_pose_pub = self.create_publisher(PoseArray, "/mcmc_pf/particle_poses", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/mcmc_pf/markers", 10)
        self.estimated_pose_pub = self.create_publisher(PoseStamped, "/mcmc_pf/estimated_pose", 10)
        self.true_pose_pub = self.create_publisher(PoseStamped, "/mcmc_pf/true_pose", 10)
        self.estimated_path_pub = self.create_publisher(NavPath, "/mcmc_pf/estimated_path", 10)
        self.true_path_pub = self.create_publisher(NavPath, "/mcmc_pf/true_path", 10)

        self.position_error_pub = self.create_publisher(Float64, "/mcmc_pf/error/position", 10)
        self.yaw_error_pub = self.create_publisher(Float64, "/mcmc_pf/error/yaw", 10)
        self.neff_pub = self.create_publisher(Float64, "/mcmc_pf/neff", 10)
        self.processing_time_pub = self.create_publisher(Float64, "/mcmc_pf/processing_time_ms", 10)
        self.mcmc_acceptance_pub = self.create_publisher(Float64, "/mcmc_pf/mcmc_acceptance_rate", 10)
        self.converged_pub = self.create_publisher(Bool, "/mcmc_pf/converged", 10)
        self.convergence_progress_pub = self.create_publisher(Float64, "/mcmc_pf/convergence_progress", 10)
        self.kidnapped_pub = self.create_publisher(Bool, "/mcmc_pf/kidnapped", 10)
        self.scan_match_score_pub = self.create_publisher(Float64, "/mcmc_pf/scan_match_score", 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self.odom_sub = self.create_subscription(
            Odometry,
            self.odom_topic,
            self.odom_callback,
            10,
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            qos_profile_sensor_data,
        )

        if not self.use_odom_as_truth:
            self.true_odom_sub = self.create_subscription(
                Odometry,
                self.true_odom_topic,
                self.true_odom_callback,
                10,
            )

            if self.enable_truth_calibration_from_initialpose:
                self.initialpose_sub = self.create_subscription(
                    PoseWithCovarianceStamped,
                    self.truth_calibration_topic,
                    self.initialpose_callback,
                    10,
                )
                self.get_logger().info(
                    f"Truth calibration enabled. Use RViz 2D Pose Estimate on "
                    f"{self.truth_calibration_topic} to align ground truth to map."
                )

        self.map_timer = self.create_timer(2.0, self.publish_map)

        self.csv_file, self.csv_writer = self.open_log_file()

        self.get_logger().info(f"Loaded map: {self.map_yaml_path}")
        self.get_logger().info(f"Map image: {self.map.image_path}")
        self.get_logger().info(f"Particles: {self.n_particles}")
        self.get_logger().info(f"Subscribing to odom: {self.odom_topic}")
        self.get_logger().info(f"Subscribing to scan: {self.scan_topic}")
        self.get_logger().info("MCMC resample-move particle filter node started.")

    def open_log_file(self):
        log_dir = os.path.expanduser(self.get_parameter("log_dir").value)
        os.makedirs(log_dir, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(log_dir, f"mcmc_pf_run_{stamp}.csv")

        f = open(path, "w", newline="")
        writer = csv.writer(f)
        writer.writerow(
            [
                "time_sec",
                "x_est",
                "y_est",
                "yaw_est",
                "x_true",
                "y_true",
                "yaw_true",
                "position_error",
                "yaw_error",
                "neff",
                "std_x",
                "std_y",
                "std_yaw",
                "converged",
                "processing_time_ms",
                "convergence_progress",
                "scan_match_score",
                "kidnap_events",
                "mcmc_acceptance_rate",
            ]
        )

        self.get_logger().info(f"Logging PF metrics to: {path}")
        return f, writer

    def publish_map(self):
        msg = self.map.to_occupancy_grid_msg(
            self.get_clock().now().to_msg(),
            self.map_frame,
        )
        self.map_pub.publish(msg)

    def odom_callback(self, msg: Odometry):
        curr_pose = odom_msg_to_pose(msg)

        if self.prev_odom_pose is None:
            self.prev_odom_pose = curr_pose
            if self.use_odom_as_truth:
                self.true_pose = curr_pose
            return

        delta = odometry_delta(self.prev_odom_pose, curr_pose)

        if self.detect_odom_kidnap(delta):
            self.handle_kidnapped("large odometry jump")
            self.prev_odom_pose = curr_pose

            if self.use_odom_as_truth:
                self.true_pose = curr_pose

            return

        self.particles = apply_odometry_motion(
            self.particles,
            delta,
            self.alphas,
            self.rng,
        )

        self.prev_odom_pose = curr_pose

        if self.use_odom_as_truth:
            self.true_pose = curr_pose

    def true_odom_callback(self, msg: Odometry):
        raw_pose = odom_msg_to_pose(msg)
        self.raw_true_pose = raw_pose
        self.true_pose = transform_truth_pose_to_map(
            raw_pose,
            self.truth_x_offset,
            self.truth_y_offset,
            self.truth_yaw_offset,
        )

    def initialpose_callback(self, msg: PoseWithCovarianceStamped):
        if self.raw_true_pose is None:
            self.get_logger().warn(
                "Received /initialpose, but no /ground_truth_odom sample has arrived yet."
            )
            return

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation

        clicked_pose = np.array(
            [p.x, p.y, quaternion_to_yaw(q)],
            dtype=np.float64,
        )

        raw = self.raw_true_pose

        yaw_offset = wrap_angle(clicked_pose[2] - raw[2])
        c = np.cos(yaw_offset)
        ss = np.sin(yaw_offset)

        x_rot = c * raw[0] - ss * raw[1]
        y_rot = ss * raw[0] + c * raw[1]

        x_offset = clicked_pose[0] - x_rot
        y_offset = clicked_pose[1] - y_rot

        self.truth_x_offset = float(x_offset)
        self.truth_y_offset = float(y_offset)
        self.truth_yaw_offset = float(yaw_offset)

        self.true_pose = transform_truth_pose_to_map(
            raw,
            self.truth_x_offset,
            self.truth_y_offset,
            self.truth_yaw_offset,
        )

        self.get_logger().info(
            "Calibrated ground truth -> map transform from RViz /initialpose:\n"
            f"  truth_x_offset   := {self.truth_x_offset:.6f}\n"
            f"  truth_y_offset   := {self.truth_y_offset:.6f}\n"
            f"  truth_yaw_offset := {self.truth_yaw_offset:.6f}"
        )


    def particle_validity_mask(self, particles):
        rows, cols, valid = self.map.world_to_map(particles[:, 0], particles[:, 1])

        legal = np.zeros(particles.shape[0], dtype=bool)
        valid_idx = np.where(valid)[0]

        if len(valid_idx) == 0:
            return legal

        r = rows[valid_idx]
        c = cols[valid_idx]

        clearance = float(self.get_parameter("mcmc_min_clearance_m").value)
        free_enough = self.map.free[r, c] & (self.map.distance_m[r, c] >= clearance)

        legal[valid_idx] = free_enough
        return legal

    def mcmc_move_particles(self, scan_msg):
        if not bool(self.get_parameter("mcmc_enabled").value):
            self.latest_mcmc_acceptance_rate = 0.0
            return

        iterations = int(self.get_parameter("mcmc_iterations").value)
        if iterations <= 0:
            self.latest_mcmc_acceptance_rate = 0.0
            return

        xy_std = float(self.get_parameter("mcmc_proposal_xy_std").value)
        yaw_std = float(self.get_parameter("mcmc_proposal_yaw_std").value)
        likelihood_scale = float(self.get_parameter("mcmc_likelihood_scale").value)

        total_accepts = 0
        total_trials = self.n_particles * iterations

        for _ in range(iterations):
            current_particles = self.particles.copy()

            proposals = current_particles.copy()
            proposals[:, 0] += self.rng.normal(0.0, xy_std, size=self.n_particles)
            proposals[:, 1] += self.rng.normal(0.0, xy_std, size=self.n_particles)
            proposals[:, 2] = wrap_angle(
                proposals[:, 2] + self.rng.normal(0.0, yaw_std, size=self.n_particles)
            )

            legal = self.particle_validity_mask(proposals)

            current_log_target = laser_log_likelihood(
                current_particles,
                scan_msg,
                self.map,
                beam_step=int(self.get_parameter("beam_step").value),
                max_beams=int(self.get_parameter("max_beams").value),
                sigma_hit=float(self.get_parameter("sigma_hit").value),
                z_hit=float(self.get_parameter("z_hit").value),
                z_rand=float(self.get_parameter("z_rand").value),
                max_range=float(self.get_parameter("max_range").value),
            )

            proposal_log_target = laser_log_likelihood(
                proposals,
                scan_msg,
                self.map,
                beam_step=int(self.get_parameter("beam_step").value),
                max_beams=int(self.get_parameter("max_beams").value),
                sigma_hit=float(self.get_parameter("sigma_hit").value),
                z_hit=float(self.get_parameter("z_hit").value),
                z_rand=float(self.get_parameter("z_rand").value),
                max_range=float(self.get_parameter("max_range").value),
            )

            # Symmetric Gaussian proposal, so q(x|x') / q(x'|x) = 1.
            # Accept if log(u) < log target(proposal) - log target(current).
            log_alpha = likelihood_scale * (proposal_log_target - current_log_target)
            log_alpha[~legal] = -np.inf

            accept = np.log(self.rng.random(self.n_particles) + 1e-300) < np.minimum(0.0, log_alpha)

            self.particles[accept] = proposals[accept]
            total_accepts += int(np.sum(accept))

        self.latest_mcmc_acceptance_rate = float(total_accepts) / float(max(1, total_trials))

        msg = Float64()
        msg.data = self.latest_mcmc_acceptance_rate
        self.mcmc_acceptance_pub.publish(msg)


    def global_reinitialize_particles(self):
        self.particles = self.map.sample_free_particles(self.n_particles, self.rng)
        self.weights = np.ones(self.n_particles, dtype=np.float64) / float(self.n_particles)

    def detect_odom_kidnap(self, delta):
        if not bool(self.get_parameter("kidnap_detection_enabled").value):
            return False

        # Only detect kidnapping after successful localization.
        if not self.converged:
            return False

        d_rot1, d_trans, d_rot2 = delta

        jump_m = float(self.get_parameter("kidnap_odom_jump_m").value)
        jump_rad = float(self.get_parameter("kidnap_odom_jump_rad").value)

        return (
            abs(float(d_trans)) > jump_m
            or abs(float(d_rot1)) > jump_rad
            or abs(float(d_rot2)) > jump_rad
        )

    def detect_scan_kidnap(self, scan_match_score):
        if not bool(self.get_parameter("kidnap_detection_enabled").value):
            return False

        # Only detect kidnapping after successful localization.
        if not self.converged:
            self.scan_match_baseline = None
            self.kidnap_bad_scan_count = 0
            return False

        if not np.isfinite(scan_match_score):
            return False

        if self.scan_match_baseline is None:
            self.scan_match_baseline = scan_match_score
            self.kidnap_bad_scan_count = 0
            return False

        required_bad = int(self.get_parameter("kidnap_required_bad_scans").value)
        drop_thresh = float(self.get_parameter("kidnap_log_likelihood_drop").value)
        abs_thresh = float(self.get_parameter("kidnap_absolute_log_likelihood").value)
        alpha = float(self.get_parameter("kidnap_baseline_alpha").value)

        likelihood_drop = self.scan_match_baseline - scan_match_score

        bad_scan = (
            likelihood_drop > drop_thresh
            or scan_match_score < abs_thresh
        )

        if bad_scan:
            self.kidnap_bad_scan_count += 1
        else:
            self.kidnap_bad_scan_count = 0
            self.scan_match_baseline = (
                (1.0 - alpha) * self.scan_match_baseline
                + alpha * scan_match_score
            )

        return self.kidnap_bad_scan_count >= required_bad

    def handle_kidnapped(self, reason):
        self.kidnap_events += 1
        self.kidnap_event_countdown = int(
            self.get_parameter("kidnap_event_latch_scans").value
        )

        self.get_logger().warn(
            f"Kidnapped robot detected: {reason}. "
            f"Reinitializing particles globally. Event #{self.kidnap_events}"
        )

        self.global_reinitialize_particles()

        self.converged = False
        self.converged_counter = 0
        self.latest_convergence_progress = 0.0
        self.scan_update_count = 0
        self.scan_match_baseline = None
        self.kidnap_bad_scan_count = 0

        # Important: respawn/teleport can cause odometry discontinuity.
        # Do not treat the next odom sample as a normal motion update.
        self.prev_odom_pose = None

        converged_msg = Bool()
        converged_msg.data = False
        self.converged_pub.publish(converged_msg)

        kidnapped_msg = Bool()
        kidnapped_msg.data = True
        self.kidnapped_pub.publish(kidnapped_msg)

    def publish_kidnap_and_scan_score(self):
        score_msg = Float64()
        score_msg.data = float(self.latest_scan_match_score)
        self.scan_match_score_pub.publish(score_msg)

        kidnapped_msg = Bool()
        kidnapped_msg.data = self.kidnap_event_countdown > 0
        self.kidnapped_pub.publish(kidnapped_msg)

        if self.kidnap_event_countdown > 0:
            self.kidnap_event_countdown -= 1


    def scan_callback(self, msg: LaserScan):
        if self.prev_odom_pose is None:
            return

        step_start_time = time.perf_counter()

        self.scan_update_count += 1

        log_prior = np.log(self.weights + 1e-300)

        log_obs = laser_log_likelihood(
            self.particles,
            msg,
            self.map,
            beam_step=int(self.get_parameter("beam_step").value),
            max_beams=int(self.get_parameter("max_beams").value),
            sigma_hit=float(self.get_parameter("sigma_hit").value),
            z_hit=float(self.get_parameter("z_hit").value),
            z_rand=float(self.get_parameter("z_rand").value),
            max_range=float(self.get_parameter("max_range").value),
        )

        self.latest_scan_match_score = float(np.max(log_obs))
        self.publish_kidnap_and_scan_score()

        if self.detect_scan_kidnap(self.latest_scan_match_score):
            self.handle_kidnapped("scan likelihood collapse")
            return

        self.weights = normalize_log_weights(log_prior + log_obs)

        estimate = weighted_pose_mean(self.particles, self.weights)
        spread = weighted_pose_spread(self.particles, self.weights, estimate)
        neff = effective_sample_size(self.weights)

        self.latest_estimate = estimate
        self.latest_spread = spread
        self.latest_neff = neff

        position_error = float("nan")
        yaw_error = float("nan")

        if self.true_pose is not None:
            position_error, yaw_error = pose_error(estimate, self.true_pose)

        progress = self.compute_convergence_progress(spread, position_error, yaw_error)
        self.latest_convergence_progress = progress

        self.publish_error_topics(position_error, yaw_error, neff, progress)
        self.update_convergence(spread, position_error, yaw_error)

        if neff < self.n_particles * float(self.get_parameter("resample_neff_fraction").value):
            self.resample_particles()

            # MCMC Metropolis-Hastings move step after resampling.
            self.mcmc_move_particles(msg)

            # After resample-move, weights are still uniform, but particles may have moved.
            estimate = weighted_pose_mean(self.particles, self.weights)
            spread = weighted_pose_spread(self.particles, self.weights, estimate)

            if self.true_pose is not None:
                position_error, yaw_error = pose_error(estimate, self.true_pose)

        stamp = self.get_clock().now().to_msg()
        self.update_paths(estimate, stamp)
        self.publish_pose_outputs(estimate, stamp)
        self.publish_visualization(estimate, spread, position_error, yaw_error, stamp)

        self.latest_processing_time_ms = (time.perf_counter() - step_start_time) * 1000.0
        self.publish_processing_time()

        self.write_log(estimate, position_error, yaw_error, neff, spread)

        if self.converged:
            self.cmd_pub.publish(Twist())

    def resample_particles(self):
        indices = systematic_resample(self.weights, self.rng)
        self.particles = self.particles[indices].copy()
        self.weights.fill(1.0 / float(self.n_particles))

        xy_std = float(self.get_parameter("roughening_xy_std").value)
        yaw_std = float(self.get_parameter("roughening_yaw_std").value)

        self.particles[:, 0] += self.rng.normal(0.0, xy_std, size=self.n_particles)
        self.particles[:, 1] += self.rng.normal(0.0, xy_std, size=self.n_particles)
        self.particles[:, 2] = wrap_angle(
            self.particles[:, 2] + self.rng.normal(0.0, yaw_std, size=self.n_particles)
        )

    def update_convergence(self, spread, position_error, yaw_error):
        xy_thresh = float(self.get_parameter("converged_std_xy").value)
        yaw_thresh = float(self.get_parameter("converged_std_yaw").value)
        required = int(self.get_parameter("converged_required_updates").value)
        min_updates = int(self.get_parameter("min_scan_updates_before_convergence").value)

        require_truth_error = bool(
            self.get_parameter("debug_require_truth_error_for_convergence").value
        )
        pos_err_thresh = float(self.get_parameter("converged_position_error").value)
        yaw_err_thresh = float(self.get_parameter("converged_yaw_error").value)

        spread_ok = (
            spread[0] < xy_thresh
            and spread[1] < xy_thresh
            and spread[2] < yaw_thresh
        )

        enough_updates = self.scan_update_count >= min_updates

        if require_truth_error:
            truth_error_ok = (
                np.isfinite(position_error)
                and np.isfinite(yaw_error)
                and position_error < pos_err_thresh
                and abs(yaw_error) < yaw_err_thresh
            )
        else:
            truth_error_ok = True

        convergence_candidate = spread_ok and enough_updates and truth_error_ok

        if convergence_candidate:
            self.converged_counter += 1
        else:
            self.converged_counter = 0

        new_converged = self.converged_counter >= required

        if new_converged and not self.converged:
            self.get_logger().info("Particle filter converged. Publishing stop command.")

        self.converged = new_converged

        msg = Bool()
        msg.data = bool(self.converged)
        self.converged_pub.publish(msg)

    def publish_processing_time(self):
        msg = Float64()
        msg.data = float(self.latest_processing_time_ms)
        self.processing_time_pub.publish(msg)

    def publish_error_topics(self, position_error, yaw_error, neff, progress):
        msg = Float64()
        msg.data = float(neff)
        self.neff_pub.publish(msg)

        msg = Float64()
        msg.data = float(progress)
        self.convergence_progress_pub.publish(msg)

        if np.isfinite(position_error):
            msg = Float64()
            msg.data = float(position_error)
            self.position_error_pub.publish(msg)

        if np.isfinite(yaw_error):
            msg = Float64()
            msg.data = float(yaw_error)
            self.yaw_error_pub.publish(msg)

    def publish_pose_outputs(self, estimate, stamp):
        est_msg = PoseStamped()
        est_msg.header.stamp = stamp
        est_msg.header.frame_id = self.map_frame
        est_msg.pose = pose_from_xyyaw(estimate[0], estimate[1], estimate[2])
        self.estimated_pose_pub.publish(est_msg)

        if self.true_pose is not None:
            true_msg = PoseStamped()
            true_msg.header.stamp = stamp
            true_msg.header.frame_id = self.map_frame
            true_msg.pose = pose_from_xyyaw(
                self.true_pose[0],
                self.true_pose[1],
                self.true_pose[2],
            )
            self.true_pose_pub.publish(true_msg)

    def compute_convergence_progress(self, spread, position_error, yaw_error):
        xy_thresh = float(self.get_parameter("converged_std_xy").value)
        yaw_thresh = float(self.get_parameter("converged_std_yaw").value)
        min_updates = int(self.get_parameter("min_scan_updates_before_convergence").value)

        pos_err_thresh = float(self.get_parameter("converged_position_error").value)
        yaw_err_thresh = float(self.get_parameter("converged_yaw_error").value)
        require_truth_error = bool(
            self.get_parameter("debug_require_truth_error_for_convergence").value
        )

        update_score = min(1.0, self.scan_update_count / max(1.0, float(min_updates)))

        spread_score = min(
            xy_thresh / max(float(spread[0]), 1e-6),
            xy_thresh / max(float(spread[1]), 1e-6),
            yaw_thresh / max(float(spread[2]), 1e-6),
            1.0,
        )

        if require_truth_error and np.isfinite(position_error) and np.isfinite(yaw_error):
            error_score = min(
                pos_err_thresh / max(float(position_error), 1e-6),
                yaw_err_thresh / max(abs(float(yaw_error)), 1e-6),
                1.0,
            )
        elif require_truth_error:
            error_score = 0.0
        else:
            error_score = 1.0

        progress = 100.0 * min(update_score, spread_score, error_score)
        return float(np.clip(progress, 0.0, 100.0))

    def append_pose_to_path(self, path_msg, pose3, stamp, last_pose):
        min_dist = float(self.get_parameter("path_publish_min_distance").value)
        max_len = int(self.get_parameter("max_path_length").value)

        if last_pose is not None:
            if np.hypot(pose3[0] - last_pose[0], pose3[1] - last_pose[1]) < min_dist:
                return last_pose

        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = self.map_frame
        ps.pose = pose_from_xyyaw(pose3[0], pose3[1], pose3[2])

        path_msg.header.stamp = stamp
        path_msg.header.frame_id = self.map_frame
        path_msg.poses.append(ps)

        if len(path_msg.poses) > max_len:
            path_msg.poses = path_msg.poses[-max_len:]

        return np.array(pose3, dtype=np.float64)

    def update_paths(self, estimate, stamp):
        self.last_estimated_path_pose = self.append_pose_to_path(
            self.estimated_path,
            estimate,
            stamp,
            self.last_estimated_path_pose,
        )
        self.estimated_path_pub.publish(self.estimated_path)

        if self.true_pose is not None:
            self.last_true_path_pose = self.append_pose_to_path(
                self.true_path,
                self.true_pose,
                stamp,
                self.last_true_path_pose,
            )
            self.true_path_pub.publish(self.true_path)


    def publish_visualization(self, estimate, spread, position_error, yaw_error, stamp):
        pose_array = PoseArray()
        pose_array.header.stamp = stamp
        pose_array.header.frame_id = self.map_frame

        max_show = min(self.n_particles, 1500)
        step = max(1, self.n_particles // max_show)

        for p in self.particles[::step]:
            pose_array.poses.append(pose_from_xyyaw(p[0], p[1], p[2]))

        self.particle_pose_pub.publish(pose_array)

        markers = MarkerArray()

        particle_marker = Marker()
        particle_marker.header.stamp = stamp
        particle_marker.header.frame_id = self.map_frame
        particle_marker.ns = "pf_particles"
        particle_marker.id = 0
        particle_marker.type = Marker.POINTS
        particle_marker.action = Marker.ADD
        particle_marker.pose.orientation.w = 1.0
        particle_marker.scale.x = 0.035
        particle_marker.scale.y = 0.035
        particle_marker.color.r = 1.0
        particle_marker.color.g = 1.0
        particle_marker.color.b = 0.0
        particle_marker.color.a = 0.45

        for p in self.particles[::step]:
            pt = Point()
            pt.x = float(p[0])
            pt.y = float(p[1])
            pt.z = 0.02
            particle_marker.points.append(pt)

        markers.markers.append(particle_marker)

        est_marker = Marker()
        est_marker.header.stamp = stamp
        est_marker.header.frame_id = self.map_frame
        est_marker.ns = "pf_estimate"
        est_marker.id = 1
        est_marker.type = Marker.ARROW
        est_marker.action = Marker.ADD
        est_marker.pose = pose_from_xyyaw(estimate[0], estimate[1], estimate[2])
        est_marker.scale.x = 0.35
        est_marker.scale.y = 0.06
        est_marker.scale.z = 0.06
        est_marker.color.r = 0.0
        est_marker.color.g = 1.0
        est_marker.color.b = 0.0
        est_marker.color.a = 1.0
        markers.markers.append(est_marker)

        if self.true_pose is not None:
            true_marker = Marker()
            true_marker.header.stamp = stamp
            true_marker.header.frame_id = self.map_frame
            true_marker.ns = "pf_true"
            true_marker.id = 2
            true_marker.type = Marker.ARROW
            true_marker.action = Marker.ADD
            true_marker.pose = pose_from_xyyaw(
                self.true_pose[0],
                self.true_pose[1],
                self.true_pose[2],
            )
            true_marker.scale.x = 0.35
            true_marker.scale.y = 0.06
            true_marker.scale.z = 0.06
            true_marker.color.r = 0.0
            true_marker.color.g = 0.35
            true_marker.color.b = 1.0
            true_marker.color.a = 1.0
            markers.markers.append(true_marker)

            error_line = Marker()
            error_line.header.stamp = stamp
            error_line.header.frame_id = self.map_frame
            error_line.ns = "pf_error"
            error_line.id = 3
            error_line.type = Marker.LINE_LIST
            error_line.action = Marker.ADD
            error_line.pose.orientation.w = 1.0
            error_line.scale.x = 0.035
            error_line.color.r = 1.0
            error_line.color.g = 0.0
            error_line.color.b = 0.0
            error_line.color.a = 1.0

            p_true = Point()
            p_true.x = float(self.true_pose[0])
            p_true.y = float(self.true_pose[1])
            p_true.z = 0.05

            p_est = Point()
            p_est.x = float(estimate[0])
            p_est.y = float(estimate[1])
            p_est.z = 0.05

            error_line.points.append(p_true)
            error_line.points.append(p_est)
            markers.markers.append(error_line)

        text = Marker()
        text.header.stamp = stamp
        text.header.frame_id = self.map_frame
        text.ns = "pf_text"
        text.id = 4
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(estimate[0])
        text.pose.position.y = float(estimate[1])
        text.pose.position.z = 0.45
        text.pose.orientation.w = 1.0
        text.scale.z = 0.16
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0

        if np.isfinite(position_error):
            text.text = (
                f"pos err: {position_error:.3f} m\n"
                f"yaw err: {yaw_error:.3f} rad\n"
                f"std: {spread[0]:.3f}, {spread[1]:.3f}, {spread[2]:.3f}\n"
                f"N_eff: {self.latest_neff:.1f}\n"
                f"converged: {self.converged}"
            )
        else:
            text.text = (
                f"std: {spread[0]:.3f}, {spread[1]:.3f}, {spread[2]:.3f}\n"
                f"N_eff: {self.latest_neff:.1f}\n"
                f"converged: {self.converged}"
            )

        markers.markers.append(text)

        progress = float(self.latest_convergence_progress)
        fill = np.clip(progress / 100.0, 0.0, 1.0)

        bar_bg = Marker()
        bar_bg.header.stamp = stamp
        bar_bg.header.frame_id = self.map_frame
        bar_bg.ns = "pf_progress_bar"
        bar_bg.id = 5
        bar_bg.type = Marker.CUBE
        bar_bg.action = Marker.ADD
        bar_bg.pose.position.x = float(estimate[0])
        bar_bg.pose.position.y = float(estimate[1] - 0.70)
        bar_bg.pose.position.z = 0.80
        bar_bg.pose.orientation.w = 1.0
        bar_bg.scale.x = 1.0
        bar_bg.scale.y = 0.08
        bar_bg.scale.z = 0.08
        bar_bg.color.r = 0.20
        bar_bg.color.g = 0.20
        bar_bg.color.b = 0.20
        bar_bg.color.a = 0.80
        markers.markers.append(bar_bg)

        bar_fill = Marker()
        bar_fill.header.stamp = stamp
        bar_fill.header.frame_id = self.map_frame
        bar_fill.ns = "pf_progress_bar"
        bar_fill.id = 6
        bar_fill.type = Marker.CUBE
        bar_fill.action = Marker.ADD
        bar_fill.pose.position.x = float(estimate[0] - 0.5 + 0.5 * fill)
        bar_fill.pose.position.y = float(estimate[1] - 0.70)
        bar_fill.pose.position.z = 0.82
        bar_fill.pose.orientation.w = 1.0
        bar_fill.scale.x = max(0.001, float(fill))
        bar_fill.scale.y = 0.09
        bar_fill.scale.z = 0.09
        bar_fill.color.r = 0.0
        bar_fill.color.g = 1.0
        bar_fill.color.b = 0.0
        bar_fill.color.a = 0.90
        markers.markers.append(bar_fill)

        progress_text = Marker()
        progress_text.header.stamp = stamp
        progress_text.header.frame_id = self.map_frame
        progress_text.ns = "pf_progress_bar"
        progress_text.id = 7
        progress_text.type = Marker.TEXT_VIEW_FACING
        progress_text.action = Marker.ADD
        progress_text.pose.position.x = float(estimate[0])
        progress_text.pose.position.y = float(estimate[1] - 0.70)
        progress_text.pose.position.z = 1.00
        progress_text.pose.orientation.w = 1.0
        progress_text.scale.z = 0.14
        progress_text.color.r = 1.0
        progress_text.color.g = 1.0
        progress_text.color.b = 1.0
        progress_text.color.a = 1.0
        progress_text.text = f"Convergence: {progress:.1f}%"
        markers.markers.append(progress_text)

        if self.kidnap_event_countdown > 0:
            kidnapped_text = Marker()
            kidnapped_text.header.stamp = stamp
            kidnapped_text.header.frame_id = self.map_frame
            kidnapped_text.ns = "pf_kidnapped_status"
            kidnapped_text.id = 8
            kidnapped_text.type = Marker.TEXT_VIEW_FACING
            kidnapped_text.action = Marker.ADD
            kidnapped_text.pose.position.x = float(estimate[0])
            kidnapped_text.pose.position.y = float(estimate[1] + 0.85)
            kidnapped_text.pose.position.z = 1.15
            kidnapped_text.pose.orientation.w = 1.0
            kidnapped_text.scale.z = 0.22
            kidnapped_text.color.r = 1.0
            kidnapped_text.color.g = 0.0
            kidnapped_text.color.b = 0.0
            kidnapped_text.color.a = 1.0
            kidnapped_text.text = "KIDNAPPED - GLOBAL RELOCALIZATION"
            markers.markers.append(kidnapped_text)

        self.marker_pub.publish(markers)

    def write_log(self, estimate, position_error, yaw_error, neff, spread):
        t = self.get_clock().now().nanoseconds * 1e-9

        if self.true_pose is None:
            true_values = [float("nan"), float("nan"), float("nan")]
        else:
            true_values = [
                float(self.true_pose[0]),
                float(self.true_pose[1]),
                float(self.true_pose[2]),
            ]

        self.csv_writer.writerow(
            [
                f"{t:.6f}",
                f"{estimate[0]:.6f}",
                f"{estimate[1]:.6f}",
                f"{estimate[2]:.6f}",
                f"{true_values[0]:.6f}",
                f"{true_values[1]:.6f}",
                f"{true_values[2]:.6f}",
                f"{position_error:.6f}",
                f"{yaw_error:.6f}",
                f"{neff:.6f}",
                f"{spread[0]:.6f}",
                f"{spread[1]:.6f}",
                f"{spread[2]:.6f}",
                int(self.converged),
                f"{self.latest_processing_time_ms:.6f}",
                f"{self.latest_convergence_progress:.6f}",
                f"{self.latest_scan_match_score:.6f}",
                int(self.kidnap_events),
                f"{self.latest_mcmc_acceptance_rate:.6f}",
            ]
        )
        self.csv_file.flush()

    def destroy_node(self):
        try:
            self.csv_file.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MCMCParticleFilterNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
