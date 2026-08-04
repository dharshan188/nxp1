import math
import time
import statistics

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Joy, LaserScan
from synapse_msgs.msg import EdgeVectors
from std_msgs.msg import String, Bool
from nav_msgs.msg import Odometry

QOS_PROFILE_DEFAULT = 10

SPEED_MIN = 0.0
SPEED_MAX = 1.0
TURN_MIN = -1.0
TURN_MAX = 1.0


# =====================================================================
# Mission Finite State Machine
# =====================================================================
class MissionState:
    NORMAL_LINE_FOLLOWING = "NORMAL_LINE_FOLLOWING"
    WAITING_FOR_SAFE_ZONE = "WAITING_FOR_SAFE_ZONE"
    ENTERING_SAFE_ZONE = "ENTERING_SAFE_ZONE"
    PARKED_IN_SAFE_ZONE = "PARKED_IN_SAFE_ZONE"
    WAITING_FOR_SERVER_ACK = "WAITING_FOR_SERVER_ACK"
    NAVIGATING_TO_NEXT_TARGET = "NAVIGATING_TO_NEXT_TARGET"
    MISSION_COMPLETE = "MISSION_COMPLETE"


class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')

        # ---------------- Parameters: line following ----------------
        self.declare_parameter('steer_sign', -1.0)
        self.declare_parameter('Kp', 0.55)
        self.declare_parameter('Ki', 0.0)
        self.declare_parameter('Kd', 0.18)
        self.declare_parameter('lookahead_blend', 0.6)
        self.declare_parameter('lane_width_px', 240.0)
        self.declare_parameter('learn_lane_width', True)
        self.declare_parameter('single_vector_side_margin', 0.20)
        self.declare_parameter('speed_straight', 0.70)
        self.declare_parameter('speed_sharp', 0.40)
        self.declare_parameter('speed_lost', 0.35)
        self.declare_parameter('steer_alpha', 0.55)
        self.declare_parameter('speed_alpha', 0.15)
        self.declare_parameter('no_vector_hold', 0.6)
        self.declare_parameter('debug_log', True)

        # ---------------- Parameters: normal obstacle avoidance ----------------
        self.declare_parameter('obstacle_enable', True)
        self.declare_parameter('obstacle_trigger_dist', 0.90)
        self.declare_parameter('obstacle_clear_dist', 1.30)
        self.declare_parameter('obstacle_fov_deg', 60.0)
        self.declare_parameter('obstacle_turn_gain', 0.9)
        self.declare_parameter('obstacle_speed', 0.32)
        self.declare_parameter('obstacle_stop_dist', 0.38)

        # --- edge-safety margin so the aim point never sits right on the curb ---
        self.declare_parameter('turn_edge_margin_px', 35.0)

        # Strict lane safety guard. Obstacle avoidance is never allowed to
        # overpower lane keeping. If the buggy drifts toward a lane edge,
        # obstacle avoidance is reduced/disabled and the controller steers
        # back into the lane first.
        self.declare_parameter('lane_guard_enable', True)
        self.declare_parameter('lane_guard_soft_error', 0.12)
        self.declare_parameter('lane_guard_hard_error', 0.22)
        self.declare_parameter('lane_guard_stop_error', 0.34)
        self.declare_parameter('lane_guard_speed_soft', 0.70)
        self.declare_parameter('lane_guard_speed_hard', 0.40)

        # --- junction lane-width ratio that triggers "wide crossing" handling ---
        self.declare_parameter('junction_width_ratio', 1.8)

        # ---------------- Parameters: straight intersection FSM ----------------
        self.declare_parameter('intersect_entry_width_ratio', 1.6)
        self.declare_parameter('intersect_exit_width_ratio', 1.25)
        self.declare_parameter('intersect_width_spike_ratio', 1.35)
        self.declare_parameter('intersect_stable_frames', 6)
        self.declare_parameter('intersect_max_time', 4.0)
        self.declare_parameter('intersect_no_vec_time', 0.18)
        self.declare_parameter('intersect_heading_gain', 0.30)
        self.declare_parameter('intersect_cte_gain', 0.20)
        self.declare_parameter('intersect_heading_blend', 0.15)
        self.declare_parameter('intersect_speed', 0.55)
        self.declare_parameter('intersect_width_samples_for_spike', 6)
        self.declare_parameter('intersect_one_vec_time', 0.25)
        self.declare_parameter('intersect_cooldown_after_timeout', 1.5)
        self.declare_parameter('intersect_max_entry_heading_deg', 30.0)
        self.declare_parameter('intersect_heading_jump_deg', 20.0)
        self.declare_parameter('intersect_memory_max_age', 1.0)
        self.declare_parameter('intersect_lock_max_heading_deg', 8.0)
        self.declare_parameter('intersect_slow_ema_alpha', 0.07)
        self.declare_parameter('intersect_fast_ema_alpha', 0.4)

        # =====================================================================
        # Safe Zone detection parameters (beam-density approach)
        # =====================================================================
        self.declare_parameter('zone_close_min_dist', 0.65)
        self.declare_parameter('zone_close_max_dist', 1.00)
        self.declare_parameter('zone_close_beam_threshold', 6)
        self.declare_parameter('zone_confirm_scans', 3)
        self.declare_parameter('zone_fov_min_deg', -45.0)
        self.declare_parameter('zone_fov_max_deg', 15.0)

        # Mission-side safe-zone LiDAR sectors.
        # IMPORTANT: in this simulator's LaserScan convention, the zone for a
        # RIGHT mission appears on NEGATIVE angles (your log shows many
        # patient-zone returns from about -170° to -60°). The zone for a
        # LEFT mission appears on POSITIVE angles. STRAIGHT keeps legacy.
        self.declare_parameter('zone_use_mission_side_sector', False)
        self.declare_parameter('zone_right_fov_min_deg', -170.0)
        self.declare_parameter('zone_right_fov_max_deg', -55.0)
        self.declare_parameter('zone_left_fov_min_deg', 55.0)
        self.declare_parameter('zone_left_fov_max_deg', 170.0)

        # =====================================================================
        # Slow approach + wall-parallel parking parameters
        # =====================================================================
        self.declare_parameter('slow_approach_speed', 0.32)
        self.declare_parameter('parking_speed', 0.25)

        # Default/fallback parking distance.
        self.declare_parameter('parking_forward_distance', 1.4)

        # NEW: target-type-specific parking distances.
        # Patient was already good with 1.2 m.
        self.declare_parameter('parking_forward_distance_patient', 1.4)
        # Hospital zone is longer, so drive deeper inside.
        self.declare_parameter('parking_forward_distance_hospital', 1.4)

        # NEW: parking/safe-zone-entry obstacle safety.
        # This remains active in ENTERING_SAFE_ZONE where old code forced turn=0.
        self.declare_parameter('parking_obstacle_enable', True)
        self.declare_parameter('parking_obstacle_fov_deg', 120.0)
        self.declare_parameter('parking_obstacle_trigger_dist', 0.85)
        self.declare_parameter('parking_obstacle_clear_dist', 1.05)
        self.declare_parameter('parking_obstacle_stop_dist', 0.38)
        self.declare_parameter('parking_obstacle_turn_gain', 1.00)
        self.declare_parameter('parking_obstacle_speed', 0.08)
        # If the buggy gets too close in parking, do not remain stuck.
        # Reverse, steer away, then creep forward again.
        self.declare_parameter('parking_reverse_speed', -0.14)
        self.declare_parameter('parking_reverse_time', 0.55)
        self.declare_parameter('parking_escape_time', 0.85)
        self.declare_parameter('parking_escape_speed', 0.08)
        self.declare_parameter('parking_escape_turn_gain', 1.00)

        self.declare_parameter('wall_min_range_m', 0.05)
        self.declare_parameter('wall_max_range_frac', 0.95)
        self.declare_parameter('wall_min_beams', 4)
        self.declare_parameter('wall_max_gap_beams', 2)
        self.declare_parameter('wall_fit_patch_deg', 50.0)
        self.declare_parameter('wall_fit_tol_m', 0.08)
        self.declare_parameter('wall_start_frames', 3)
        self.declare_parameter('wall_center_deadband_m', 0.05)
        self.declare_parameter('wall_align_deadband_deg', 6.0)
        self.declare_parameter('wall_align_gain', 0.6)
        self.declare_parameter('wall_align_sign', -1.0)
        self.declare_parameter('wall_hold_frames', 5)
        self.declare_parameter('wall_confirm_frames', 2)
        self.declare_parameter('parking_timeout_s', 25.0)

        # =====================================================================
        # Mission synchronization parameters
        # =====================================================================
        self.declare_parameter('target_type_wait_timeout', 0.5)

        self._reload_params()

        # ---------------- Runtime state ----------------
        self.error = 0.0
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = None
        self.vectors_available = False
        self.last_vector_time = None
        self.last_single_side = 'LEFT'
        self.learned_lane_width = self.lane_width_px
        self.target_turn = 0.0
        self.target_speed = 0.0
        self.filtered_turn = 0.0
        self.filtered_speed = 0.0
        self.last_good_turn = 0.0

        self.obstacle_detected = False
        self.obstacle_turn = 0.0
        self.nearest_dist = float('inf')

        # NEW: parking-specific obstacle runtime state.
        self.parking_obstacle_detected = False
        self.parking_obstacle_stop = False
        self.parking_obstacle_turn = 0.0
        self.parking_front_min = float('inf')
        self.parking_front_left_min = float('inf')
        self.parking_front_right_min = float('inf')
        self._parking_recovery_phase = "NONE"   # NONE / REVERSING / ESCAPING
        self._parking_recovery_until = 0.0
        self._parking_escape_turn = 0.0

        self._tick = 0
        self.current_mission = "STRAIGHT"
        self._straight_junction_side = None

        # Straight-intersection state-machine runtime state
        self._in_intersection = False
        self._intersection_entry_time = None
        self._intersection_heading = 0.0
        self._intersection_cte = 0.0
        self._intersection_stable_count = 0
        self._width_ema = self.lane_width_px
        self._width_ema_samples = 0
        self._last_good_heading = 0.0
        self._last_good_cte = 0.0
        self._heading_ema_fast = 0.0
        self._heading_ema_slow = 0.0
        self._heading_ema_init = False
        self._last_two_vec_time = None
        self._intersect_cooldown_until = 0.0

        # =====================================================================
        # Mission Finite State Machine
        # =====================================================================
        self.mission_state = MissionState.NORMAL_LINE_FOLLOWING
        self.target_qr_string = ""
        self.last_valid_mission = "NONE"

        # Beam-density safe-zone detection bookkeeping
        self._zone_close_count = 0
        self._zone_total_valid = 0
        self._zone_min_dist = float('inf')
        self._zone_consecutive_scans = 0
        self._zone_detection_state = "MISS"
        self._safe_zone_published = False

        # Parking bookkeeping
        self._wall_geom = {}
        self._wall_side = None
        self._wall_phase = "SEARCHING"
        self._wall_present_count = 0
        self._wall_miss_count = 0
        self._wall_start_dist = 0.0
        self._wall_last_s = None
        self._wall_align_ok = 0
        self._wall_align_error = 0.0
        self._wall_align_valid = False
        self._wall_cross_armed = False
        self._wall_cross_sign = 0
        self._wall_last_align = 0.0
        self._wall_len_m = 0.0
        self._wall_lateral_m = 0.0
        self._wall_midpoint_dist = 0.0
        self._park_start_time = 0.0
        self._park_start_dist = 0.0
        self._last_travel_at_check = None
        self._parking_log_time = 0.0
        self._scan_seq = 0
        self._park_check_seq = -1

        # Travelled distance
        self._travel_dist = 0.0
        self._last_travel_time = None
        self._last_cmd_speed = 0.0
        self._odom_linear_x = None
        self._odom_time = None

        # LiDAR orientation-map debug
        self._orient_map_log_time = 0.0
        self._last_ranges = []
        self._last_range_count = 0
        self._last_angle_min = 0.0
        self._last_angle_increment = 1.0

        # Mission synchronization state
        self.pending_target_type = None
        self.pending_target_qr = None
        self.pending_target_qr_time = None
        self.mission_ready = False
        self.active_target_type = None
        self.active_target_qr = ""
        self._last_mission_key = None
        self.pending_next_type = None
        self.pending_next_qr = None
        self.pending_next_qr_time = None

        # LiDAR zone analysis results
        self._zone_valid_distances = []
        self._zone_sector_data = []
        self._zone_active_fov_min_deg = self.zone_fov_min_deg
        self._zone_active_fov_max_deg = self.zone_fov_max_deg

        # Timers for periodic log messages
        self._wait_log_time = 0.0
        self._lidar_debug_time = 0.0

        # ---------------- ROS plumbing ----------------
        self.create_subscription(EdgeVectors, '/edge_vectors', self.edge_vectors_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(String, '/mission/turn', self.mission_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(String, '/target_qr', self.target_qr_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(String, '/target_type', self.target_type_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(String, '/resume_line_following', self.resume_callback, 10)
        self.create_subscription(String, '/mission/available', self.mission_available_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, QOS_PROFILE_DEFAULT)

        self.pub_joy = self.create_publisher(Joy, '/cerebri/in/joy', QOS_PROFILE_DEFAULT)
        self.pub_safe_zone = self.create_publisher(Bool, '/safe_zone', QOS_PROFILE_DEFAULT)

        self.create_timer(0.033, self.control_loop)
        self.create_timer(1.0, self._reload_params)

        self.get_logger().info(
            "Lane-following controller loaded.\n"
            f"Mission FSM initial state: {self.mission_state}"
        )

    def _transition_mission_state(self, new_state, reason=""):
        old = self.mission_state
        if old == new_state:
            return

        self.mission_state = new_state

        if new_state in (MissionState.NORMAL_LINE_FOLLOWING,
                         MissionState.NAVIGATING_TO_NEXT_TARGET):
            self._promote_pending_next()

        if self.mission_state != new_state:
            self.get_logger().info(
                (f"MISSION STATE: {old} → {new_state} → {self.mission_state}  ({reason})"
                 if reason else f"MISSION STATE: {old} → {new_state} → {self.mission_state}")
            )
        else:
            self.get_logger().info(
                (f"MISSION STATE: {old} → {new_state}  ({reason})"
                 if reason else f"MISSION STATE: {old} → {new_state}")
            )

    # =====================================================================
    # Safe-zone detection reset
    # =====================================================================
    def _reset_zone_detection(self):
        self._zone_close_count = 0
        self._zone_total_valid = 0
        self._zone_min_dist = float('inf')
        self._zone_consecutive_scans = 0
        self._zone_detection_state = "MISS"
        self._safe_zone_published = False

        self._wall_side = None
        self._wall_phase = "SEARCHING"
        self._wall_present_count = 0
        self._wall_miss_count = 0
        self._wall_start_dist = 0.0
        self._wall_last_s = None
        self._wall_align_ok = 0
        self._wall_align_error = 0.0
        self._wall_align_valid = False
        self._wall_cross_armed = False
        self._wall_cross_sign = 0
        self._wall_last_align = 0.0
        self._wall_midpoint_dist = 0.0
        self._park_start_time = 0.0
        self._park_start_dist = 0.0
        self._last_travel_at_check = None
        self._travel_dist = 0.0
        self._last_travel_time = None
        self._zone_valid_distances = []
        self._zone_sector_data = []

        self.parking_obstacle_detected = False
        self.parking_obstacle_stop = False
        self.parking_obstacle_turn = 0.0
        self.parking_front_min = float('inf')
        self.parking_front_left_min = float('inf')
        self.parking_front_right_min = float('inf')
        self._parking_recovery_phase = "NONE"
        self._parking_recovery_until = 0.0
        self._parking_escape_turn = 0.0

    # ------------------------------------------------------------------
    # Parameter reload
    # ------------------------------------------------------------------
    def _reload_params(self):
        g = lambda n: self.get_parameter(n).value

        self.steer_sign = float(g('steer_sign'))
        self.Kp = g('Kp')
        self.Ki = g('Ki')
        self.Kd = g('Kd')
        self.lookahead_blend = g('lookahead_blend')
        self.lane_width_px = g('lane_width_px')
        self.learn_lane_width = g('learn_lane_width')
        self.side_margin = g('single_vector_side_margin')
        self.speed_straight = g('speed_straight')
        self.speed_sharp = g('speed_sharp')
        self.speed_lost = g('speed_lost')
        self.steer_alpha = g('steer_alpha')
        self.speed_alpha = g('speed_alpha')
        self.no_vector_hold = g('no_vector_hold')
        self.debug_log = g('debug_log')

        self.obstacle_enable = g('obstacle_enable')
        self.obstacle_trigger_dist = g('obstacle_trigger_dist')
        self.obstacle_clear_dist = g('obstacle_clear_dist')
        self.obstacle_fov_deg = g('obstacle_fov_deg')
        self.obstacle_turn_gain = g('obstacle_turn_gain')
        self.obstacle_speed = g('obstacle_speed')
        self.obstacle_stop_dist = g('obstacle_stop_dist')

        self.turn_edge_margin_px = g('turn_edge_margin_px')
        self.lane_guard_enable = g('lane_guard_enable')
        self.lane_guard_soft_error = g('lane_guard_soft_error')
        self.lane_guard_hard_error = g('lane_guard_hard_error')
        self.lane_guard_stop_error = g('lane_guard_stop_error')
        self.lane_guard_speed_soft = g('lane_guard_speed_soft')
        self.lane_guard_speed_hard = g('lane_guard_speed_hard')
        self.junction_width_ratio = g('junction_width_ratio')

        self.intersect_entry_width_ratio = g('intersect_entry_width_ratio')
        self.intersect_exit_width_ratio = g('intersect_exit_width_ratio')
        self.intersect_width_spike_ratio = g('intersect_width_spike_ratio')
        self.intersect_stable_frames = int(g('intersect_stable_frames'))
        self.intersect_max_time = g('intersect_max_time')
        self.intersect_no_vec_time = g('intersect_no_vec_time')
        self.intersect_heading_gain = g('intersect_heading_gain')
        self.intersect_cte_gain = g('intersect_cte_gain')
        self.intersect_heading_blend = g('intersect_heading_blend')
        self.intersect_speed = g('intersect_speed')
        self.intersect_width_samples_for_spike = int(g('intersect_width_samples_for_spike'))
        self.intersect_one_vec_time = g('intersect_one_vec_time')
        self.intersect_cooldown_after_timeout = g('intersect_cooldown_after_timeout')
        self.intersect_max_entry_heading_deg = g('intersect_max_entry_heading_deg')
        self.intersect_heading_jump_deg = g('intersect_heading_jump_deg')
        self.intersect_memory_max_age = g('intersect_memory_max_age')
        self.intersect_lock_max_heading_deg = g('intersect_lock_max_heading_deg')
        self.intersect_slow_ema_alpha = g('intersect_slow_ema_alpha')
        self.intersect_fast_ema_alpha = g('intersect_fast_ema_alpha')

        self.zone_close_min_dist = g('zone_close_min_dist')
        self.zone_close_max_dist = g('zone_close_max_dist')
        self.zone_close_beam_threshold = int(g('zone_close_beam_threshold'))
        self.zone_confirm_scans = int(g('zone_confirm_scans'))
        self.zone_fov_min_deg = g('zone_fov_min_deg')
        self.zone_fov_max_deg = g('zone_fov_max_deg')
        self.zone_use_mission_side_sector = g('zone_use_mission_side_sector')
        self.zone_right_fov_min_deg = g('zone_right_fov_min_deg')
        self.zone_right_fov_max_deg = g('zone_right_fov_max_deg')
        self.zone_left_fov_min_deg = g('zone_left_fov_min_deg')
        self.zone_left_fov_max_deg = g('zone_left_fov_max_deg')

        self.slow_approach_speed = g('slow_approach_speed')
        self.parking_speed = g('parking_speed')
        self.parking_forward_distance = g('parking_forward_distance')
        self.parking_forward_distance_patient = g('parking_forward_distance_patient')
        self.parking_forward_distance_hospital = g('parking_forward_distance_hospital')

        self.parking_obstacle_enable = g('parking_obstacle_enable')
        self.parking_obstacle_fov_deg = g('parking_obstacle_fov_deg')
        self.parking_obstacle_trigger_dist = g('parking_obstacle_trigger_dist')
        self.parking_obstacle_clear_dist = g('parking_obstacle_clear_dist')
        self.parking_obstacle_stop_dist = g('parking_obstacle_stop_dist')
        self.parking_obstacle_turn_gain = g('parking_obstacle_turn_gain')
        self.parking_obstacle_speed = g('parking_obstacle_speed')
        self.parking_reverse_speed = g('parking_reverse_speed')
        self.parking_reverse_time = g('parking_reverse_time')
        self.parking_escape_time = g('parking_escape_time')
        self.parking_escape_speed = g('parking_escape_speed')
        self.parking_escape_turn_gain = g('parking_escape_turn_gain')

        self.wall_min_range_m = g('wall_min_range_m')
        self.wall_max_range_frac = g('wall_max_range_frac')
        self.wall_min_beams = int(g('wall_min_beams'))
        self.wall_max_gap_beams = int(g('wall_max_gap_beams'))
        self.wall_fit_patch_deg = g('wall_fit_patch_deg')
        self.wall_fit_tol_m = g('wall_fit_tol_m')
        self.wall_start_frames = int(g('wall_start_frames'))
        self.wall_center_deadband_m = g('wall_center_deadband_m')
        self.wall_align_deadband_deg = g('wall_align_deadband_deg')
        self.wall_align_gain = g('wall_align_gain')
        self.wall_align_sign = g('wall_align_sign')
        self.wall_hold_frames = int(g('wall_hold_frames'))
        self.wall_confirm_frames = int(g('wall_confirm_frames'))
        self.parking_timeout_s = g('parking_timeout_s')

        self.target_type_wait_timeout = g('target_type_wait_timeout')

    # ------------------------------------------------------------------
    # Mission synchronization callbacks
    # ------------------------------------------------------------------
    def target_type_callback(self, msg):
        if not msg.data or not msg.data.strip():
            return

        value = msg.data.strip().upper()
        if value not in ("PATIENT", "HOSPITAL"):
            self.get_logger().info(f"Ignoring unknown /target_type value: {msg.data}")
            return

        if self._mission_in_progress():
            if self.pending_next_qr is None and self.pending_next_type is None:
                self.pending_next_type = value
                self.get_logger().info(
                    f"New assignment queued while {self.mission_state}: target type {value} (pending)."
                )
            elif self.pending_next_type == value:
                self.get_logger().info(f"Duplicate /target_type ignored: {value}")
            else:
                self.pending_next_type = value
                self.get_logger().warn(
                    f"Pending target type replaced with {value} (current mission still active)."
                )
            return

        if self.pending_target_qr is not None and self.pending_target_type is None:
            self.pending_target_type = value
            self.get_logger().info(
                f"Target type received: {value} — matches pending QR {self.pending_target_qr}; mission ready."
            )
            self.mission_ready = True
            self._activate_pending_mission()
            return

        if self.pending_target_type == value:
            self.get_logger().info(f"Duplicate /target_type ignored: {value}")
            return

        if self.pending_target_type is None:
            self.pending_target_type = value
            self.get_logger().info(
                f"Target type received: {value} — awaiting /target_qr to activate mission."
            )
        else:
            self.pending_target_type = value
            self.get_logger().warn(f"Orphaned pending target type replaced with {value}.")

    def target_qr_callback(self, msg):
        if not msg.data or not msg.data.strip():
            return

        qr = msg.data.strip()

        if self._mission_in_progress():
            if qr == self.active_target_qr:
                self.get_logger().info(f"Duplicate /target_qr ignored (active mission: {qr}).")
                if self.pending_next_qr is None and self.pending_next_type == self.active_target_type:
                    self.pending_next_type = None
                return

            if self.pending_next_qr == qr:
                self.get_logger().info(f"Duplicate queued QR ignored: {qr}")
                return

            self.pending_next_qr = qr
            self.pending_next_qr_time = time.time()
            self.get_logger().info(
                f"New assignment queued while {self.mission_state}: QR {qr} (pending)."
            )
            return

        if self.pending_target_type is not None and self.pending_target_qr is None:
            self.pending_target_qr = qr
            self.pending_target_qr_time = time.time()
            self.get_logger().info(
                f"Target QR received: {qr} — matches pending type {self.pending_target_type}; mission ready."
            )
            self.mission_ready = True
            self._activate_pending_mission()
            return

        if self.pending_target_qr == qr:
            self.get_logger().info(f"Duplicate /target_qr ignored: {qr}")
            return

        if self.pending_target_qr is None:
            self.pending_target_qr = qr
            self.pending_target_qr_time = time.time()
            self.get_logger().info(
                f"Target QR received before /target_type: {qr} — awaiting /target_type "
                f"(legacy fallback after {self.target_type_wait_timeout:.1f}s)."
            )
        else:
            self.pending_target_qr = qr
            self.pending_target_qr_time = time.time()
            self.get_logger().warn(f"Pending QR replaced with {qr} (awaiting its /target_type).")

    # =====================================================================
    # Mission synchronization helpers
    # =====================================================================
    def _mission_in_progress(self):
        return self.mission_state in (
            MissionState.WAITING_FOR_SAFE_ZONE,
            MissionState.ENTERING_SAFE_ZONE,
            MissionState.PARKED_IN_SAFE_ZONE,
            MissionState.WAITING_FOR_SERVER_ACK,
            MissionState.MISSION_COMPLETE,
        )

    def _clear_pending_assembly(self):
        self.pending_target_type = None
        self.pending_target_qr = None
        self.pending_target_qr_time = None
        self.mission_ready = False

    def _activate_pending_mission(self, legacy=False):
        ttype = "UNKNOWN" if legacy else self.pending_target_type
        qr = self.pending_target_qr

        if qr is None:
            self._clear_pending_assembly()
            return

        if ttype is None:
            ttype = "UNKNOWN"

        key = (ttype, qr)
        if key == self._last_mission_key:
            self.get_logger().info(f"Duplicate mission ignored (already processed): type={ttype}, QR={qr}")
            self._clear_pending_assembly()
            return

        self._last_mission_key = key
        self.active_target_type = ttype
        self.active_target_qr = qr
        self.target_qr_string = qr

        self.get_logger().info(
            "Mission cached:\n"
            f"Target Type : {ttype}\n"
            f"QR          : {qr}"
        )

        self._clear_pending_assembly()
        self._reset_zone_detection()
        self._transition_mission_state(
            MissionState.WAITING_FOR_SAFE_ZONE,
            f"/target_qr received: {qr} (target type: {ttype})"
        )
        self.get_logger().info("Mission activated.")

    def _complete_active_mission(self, reason=""):
        if self.active_target_type is not None:
            self.get_logger().info(
                "Mission completed.\n"
                f"Reason        : {reason}\n"
                f"Target Type   : {self.active_target_type}\n"
                f"QR            : {self.active_target_qr}"
            )

        self.active_target_type = None
        self.active_target_qr = ""
        self.target_qr_string = ""
        self._last_mission_key = None
        self._reset_zone_detection()

    def _promote_pending_next(self):
        if self.pending_next_type is None and self.pending_next_qr is None:
            return

        self.pending_target_type = self.pending_next_type
        self.pending_target_qr = self.pending_next_qr
        self.pending_target_qr_time = self.pending_next_qr_time
        self.pending_next_type = None
        self.pending_next_qr = None
        self.pending_next_qr_time = None

        self.get_logger().info("Pending mission promoted.")

        if self.pending_target_type is not None and self.pending_target_qr is not None:
            self.mission_ready = True
            self._activate_pending_mission()

    # ------------------------------------------------------------------
    # Resume Line Following callback
    # ------------------------------------------------------------------
    def resume_callback(self, msg):
        received = msg.data.strip() if msg.data else ""
        if received not in ("RESUME", "MISSION_COMPLETE"):
            self.get_logger().info(f"Ignoring resume message: {received}")
            return
        self._process_resume(received)

    def _process_resume(self, received):
        if self.mission_state not in (MissionState.PARKED_IN_SAFE_ZONE,
                                      MissionState.WAITING_FOR_SERVER_ACK):
            self.get_logger().info(f"{received} ignored in state {self.mission_state}")
            return

        if received == "MISSION_COMPLETE":
            self._complete_active_mission("MISSION_COMPLETE received")
            self._transition_mission_state(
                MissionState.MISSION_COMPLETE,
                "Server signalled mission complete"
            )
            self.get_logger().info(
                "====================================\n"
                "Waiting For New Goal Assignment...\n"
                "===================================="
            )
            return

        self._complete_active_mission("RESUME received")
        self._transition_mission_state(
            MissionState.NAVIGATING_TO_NEXT_TARGET,
            "Server ACK / RESUME received"
        )
        self._wait_log_time = 0.0

    # ------------------------------------------------------------------
    # Mission topic callback
    # ------------------------------------------------------------------
    def mission_callback(self, msg):
        mission = msg.data

        if not mission or not mission.strip() or mission.strip() == "NONE":
            self.get_logger().info("Mission Topic Received: NONE — ignoring")
            return

        mission = mission.strip()
        if mission not in ["LEFT", "RIGHT", "STRAIGHT"]:
            return

        self.get_logger().info(f"Mission Topic Received: {mission}")

        if self.mission_state == MissionState.MISSION_COMPLETE:
            self.get_logger().info(
                f"Mission Topic ignored in MISSION_COMPLETE: {mission} (waiting for /mission/available)"
            )
            return

        if self.mission_state == MissionState.WAITING_FOR_SERVER_ACK:
            if mission == self.last_valid_mission:
                self.get_logger().info("Ignoring duplicate mission")
                return

            self.get_logger().info("Assignment Accepted — Resuming Navigation")
            self._reset_zone_detection()
            self._complete_active_mission("New mission direction")
            self._transition_mission_state(
                MissionState.NAVIGATING_TO_NEXT_TARGET,
                "New mission direction while waiting for server"
            )
            self.last_valid_mission = mission
            self.current_mission = mission
            self._straight_junction_side = None
            self._reset_intersection_state()
            self._heading_ema_init = False
            return

        if self.mission_state == MissionState.WAITING_FOR_SAFE_ZONE:
            if mission != self.current_mission:
                self.get_logger().info(f"Mission changed to {mission} while WAITING_FOR_SAFE_ZONE")
                self.current_mission = mission
                self._straight_junction_side = None
                self._reset_intersection_state()
                self._heading_ema_init = False
            self.last_valid_mission = mission
            return

        if mission != self.current_mission:
            self.get_logger().info(f"Mission changed to {mission}")
            self.current_mission = mission
            self._straight_junction_side = None
            self._reset_intersection_state()
            self._heading_ema_init = False

        self.last_valid_mission = mission

    # ------------------------------------------------------------------
    # New Mission Available callback
    # ------------------------------------------------------------------
    def mission_available_callback(self, msg):
        if not msg.data or not msg.data.strip():
            return

        assignment = msg.data.strip()
        upper = assignment.upper()

        if not (upper.startswith("PATIENT_") or upper.startswith("HOSPITAL_")):
            self.get_logger().info(f"/mission/available ignored: invalid target payload: {assignment}")
            return

        if self.mission_state in (
                MissionState.MISSION_COMPLETE,
                MissionState.PARKED_IN_SAFE_ZONE,
                MissionState.WAITING_FOR_SERVER_ACK):
            self._complete_active_mission("New mission assigned by QR Detector")
            self._transition_mission_state(
                MissionState.NAVIGATING_TO_NEXT_TARGET,
                f"New mission assigned by QR Detector ({assignment})"
            )
            return

        self.get_logger().info(f"/mission/available ignored in state {self.mission_state}")

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _aim_x(self, vector):
        p0, p1 = vector[0], vector[1]
        near, far = (p1, p0) if p1.y >= p0.y else (p0, p1)
        b = self.lookahead_blend
        return (1.0 - b) * near.x + b * far.x

    @staticmethod
    def _mean_x(vector):
        return (vector[0].x + vector[1].x) / 2.0

    def _clamped_offset(self, offset, lane_width):
        margin = self.turn_edge_margin_px
        if lane_width <= 2.0 * margin:
            return lane_width / 2.0
        return max(margin, min(lane_width - margin, offset))

    def _clamp_to_lane(self, lane_center, left_edge, right_edge):
        margin = self.turn_edge_margin_px
        if right_edge - left_edge <= 2.0 * margin:
            return 0.5 * (left_edge + right_edge)
        return max(left_edge + margin, min(right_edge - margin, lane_center))

    # ------------------------------------------------------------------
    # Intersection helpers
    # ------------------------------------------------------------------
    def _reset_intersection_state(self):
        self._in_intersection = False
        self._intersection_entry_time = None
        self._intersection_heading = 0.0
        self._intersection_cte = 0.0
        self._intersection_stable_count = 0

    def _in_cooldown(self, now):
        return now < self._intersect_cooldown_until

    @staticmethod
    def _vector_heading(vector):
        p0, p1 = vector[0], vector[1]
        near, far = (p1, p0) if p1.y >= p0.y else (p0, p1)
        dx = far.x - near.x
        dy = far.y - near.y
        return math.atan2(dx, -dy)

    @staticmethod
    def _lane_heading(vec_left, vec_right):
        return 0.5 * (LineFollower._vector_heading(vec_left)
                      + LineFollower._vector_heading(vec_right))

    @staticmethod
    def _clamp_heading(h, max_deg=45.0):
        lim = math.radians(max_deg)
        if abs(h) > lim:
            return max(-lim, min(lim, h)), False
        return h, True

    def _enter_intersection(self, heading, cte, reason):
        lim_hard = math.radians(self.intersect_max_entry_heading_deg)
        lim_lock = math.radians(self.intersect_lock_max_heading_deg)
        clamped_hard = False

        if abs(heading) > lim_hard:
            self.get_logger().warn(
                f"Intersection heading {math.degrees(heading):+.1f}deg implausible; "
                f"clamping to 0 (straight). reason={reason}"
            )
            heading = 0.0
            clamped_hard = True
        elif abs(heading) > lim_lock:
            heading = math.copysign(lim_lock, heading)

        self._in_intersection = True
        self._intersection_entry_time = time.time()
        self._intersection_heading = float(heading)
        self._intersection_cte = float(cte)
        self._intersection_stable_count = 0
        self.integral = 0.0
        self.prev_time = None

        self.get_logger().info(
            f"*** Entering STRAIGHT_INTERSECTION reason={reason} "
            f"heading={math.degrees(heading):+.1f}deg cte={cte:+.2f}"
            f"{' (clamped)' if clamped_hard else ''}"
        )

    def _exit_intersection(self, reason):
        self.get_logger().info(
            f"*** Exiting STRAIGHT_INTERSECTION reason={reason} stable={self._intersection_stable_count}"
        )
        self.integral = 0.0
        self.prev_time = None

        if reason in ("timeout", "watchdog"):
            self._last_good_heading = 0.0
            self._last_good_cte = 0.0
            self.last_good_turn = 0.0
            self._heading_ema_init = False
            self._intersect_cooldown_until = time.time() + self.intersect_cooldown_after_timeout

        self._reset_intersection_state()

    # ------------------------------------------------------------------
    # Edge vectors callback
    # ------------------------------------------------------------------
    def edge_vectors_callback(self, message):
        now = time.time()
        img_w = float(message.image_width)
        img_center = img_w / 2.0
        if img_center <= 0:
            return

        count = message.vector_count
        mission_straight = (self.current_mission == "STRAIGHT")

        if not mission_straight and self._in_intersection:
            self._reset_intersection_state()

        if self._in_intersection and self._intersection_entry_time is not None:
            if (now - self._intersection_entry_time) > self.intersect_max_time:
                self._exit_intersection("timeout")

        # ---------------------------------------------------------------
        # Case A: two vectors seen
        # ---------------------------------------------------------------
        if count >= 2:
            v1, v2 = message.vector_1, message.vector_2
            xa = self._aim_x(v1)
            xb = self._aim_x(v2)

            if self._mean_x(v1) < self._mean_x(v2):
                left_x, right_x = xa, xb
                vec_left, vec_right = v1, v2
            else:
                left_x, right_x = xb, xa
                vec_left, vec_right = v2, v1

            lane_width = right_x - left_x
            current_heading = self._lane_heading(vec_left, vec_right)

            if mission_straight:
                if self._in_intersection:
                    width_ok = (lane_width <= self.learned_lane_width * self.intersect_exit_width_ratio
                                and lane_width > 0)
                    if width_ok:
                        self._intersection_stable_count += 1
                        blend = self.intersect_heading_blend
                        self._intersection_heading = ((1.0 - blend) * self._intersection_heading
                                                      + blend * current_heading)
                        new_cte = ((left_x + right_x) * 0.5 - img_center) / img_center
                        self._intersection_cte = ((1.0 - blend) * self._intersection_cte
                                                  + blend * new_cte)
                    else:
                        self._intersection_stable_count = 0

                    if self._intersection_stable_count >= self.intersect_stable_frames:
                        self._exit_intersection("recovery")

                        # STRAIGHT means STRAIGHT: never choose the left/right branch
                        # after an intersection recovery. The old code selected a
                        # junction side (L/R), which could make a STRAIGHT goal turn
                        # left. Keep the aim at the lane midpoint only.
                        lane_center = (left_x + right_x) / 2.0
                        self._straight_junction_side = None

                        self._update_lane_width_memory(lane_width, img_w)
                        self._last_two_vec_time = now
                        self._last_good_heading = current_heading
                        lane_center = self._clamp_to_lane(lane_center, left_x, right_x)
                        raw_cte = (lane_center - img_center) / img_center
                        self._last_good_cte = raw_cte
                        self.vectors_available = True
                        self.last_vector_time = now
                        self._apply_line_pid(raw_cte, now)
                        return

                    self.vectors_available = True
                    self.last_vector_time = now
                    cte = self._intersection_cte
                    turn = (self.intersect_heading_gain * self._intersection_heading
                            + self.intersect_cte_gain * cte)
                    self.target_turn = max(TURN_MIN, min(TURN_MAX, turn))
                    self.target_speed = self.intersect_speed
                    return

                self._last_two_vec_time = now
                self._update_lane_width_memory(lane_width, img_w)

                # STRAIGHT mission fix:
                # Do not bias toward left or right at a wide junction.
                # The previous L/R side-selection made STRAIGHT sometimes
                # turn left. Use only the lane midpoint and let the
                # intersection lock drive straight through.
                lane_center = (left_x + right_x) / 2.0
                self._straight_junction_side = None

                lane_center = self._clamp_to_lane(lane_center, left_x, right_x)
                raw_cte = (lane_center - img_center) / img_center

                reason = None
                if lane_width > self.learned_lane_width * self.intersect_entry_width_ratio:
                    reason = "wide"
                elif (self._width_ema_samples >= self.intersect_width_samples_for_spike
                      and self._width_ema > 0
                      and lane_width > self._width_ema * self.intersect_width_spike_ratio):
                    reason = "spike"

                if reason is not None:
                    # STRAIGHT intersection lock fix:
                    # At a wide/spike junction, do not use a biased CTE or
                    # noisy lane heading that can pull the buggy left/right.
                    # Lock to straight heading and zero CTE until normal lane
                    # width returns.
                    self._last_good_heading = 0.0
                    self._last_good_cte = 0.0
                    self._enter_intersection(0.0, 0.0, reason)
                    self.vectors_available = True
                    self.last_vector_time = now
                    self.target_turn = 0.0
                    self.target_speed = self.intersect_speed
                    self.last_good_turn = 0.0
                    return

                self._last_good_heading = current_heading
                self._last_good_cte = raw_cte
                self._update_heading_ema(current_heading)
                self.vectors_available = True
                self._apply_line_pid(raw_cte, now)
                return

            # LEFT / RIGHT mission with two vectors
            if self.current_mission == "LEFT":
                offset = self._clamped_offset(0.35 * lane_width, lane_width)
                lane_center = left_x + offset
            elif self.current_mission == "RIGHT":
                offset = self._clamped_offset(0.35 * lane_width, lane_width)
                lane_center = right_x - offset
            else:
                lane_center = (left_x + right_x) / 2.0

            lane_center = self._clamp_to_lane(lane_center, left_x, right_x)
            self.vectors_available = True
            self._update_lane_width_memory(lane_width, img_w)

        # ---------------------------------------------------------------
        # Case B: exactly one vector
        # ---------------------------------------------------------------
        elif count == 1:
            v = message.vector_1
            aim = self._aim_x(v)
            mean_x = self._mean_x(v)
            band = self.side_margin * img_center

            if mean_x < img_center - band:
                self.last_single_side = 'LEFT'
            elif mean_x > img_center + band:
                self.last_single_side = 'RIGHT'

            lane_width = self.learned_lane_width
            one_vec_heading = self._vector_heading(v)

            if mission_straight:
                if self._in_intersection:
                    self.vectors_available = True
                    self.last_vector_time = now
                    self._intersection_stable_count = 0
                    cte = self._intersection_cte
                    turn = (self.intersect_heading_gain * self._intersection_heading
                            + self.intersect_cte_gain * cte)
                    self.target_turn = max(TURN_MIN, min(TURN_MAX, turn))
                    self.target_speed = self.intersect_speed
                    return

                jump_lim = math.radians(self.intersect_heading_jump_deg)
                have_heading_ref = self._heading_ema_init
                heading_jump = abs(one_vec_heading - self._heading_ema_fast) if have_heading_ref else 0.0

                if have_heading_ref and not self._in_cooldown(now) and heading_jump > jump_lim:
                    snap_heading = self._heading_ema_slow
                    snap_cte = self.error
                    self._enter_intersection(
                        snap_heading, snap_cte,
                        f"one_vec_jump({math.degrees(heading_jump):.0f}deg)"
                    )
                    self.vectors_available = True
                    self.last_vector_time = now
                    cte = self._intersection_cte
                    turn = (self.intersect_heading_gain * self._intersection_heading
                            + self.intersect_cte_gain * cte)
                    self.target_turn = max(TURN_MIN, min(TURN_MAX, turn))
                    self.target_speed = self.intersect_speed
                    self.last_good_turn = self.target_turn
                    return

            if self.current_mission == "LEFT":
                offset = self._clamped_offset(0.40 * lane_width, lane_width)
            elif self.current_mission == "RIGHT":
                offset = self._clamped_offset(0.60 * lane_width, lane_width)
            else:
                offset = self._clamped_offset(0.50 * lane_width, lane_width)

            if self.last_single_side == 'LEFT':
                lane_center = aim + offset
                lane_center = self._clamp_to_lane(lane_center, aim, aim + lane_width)
            else:
                lane_center = aim - (lane_width - offset)
                lane_center = self._clamp_to_lane(lane_center, aim - lane_width, aim)

            self.vectors_available = True

        # ---------------------------------------------------------------
        # Case C: no vectors
        # ---------------------------------------------------------------
        else:
            if mission_straight and self._in_intersection:
                self.vectors_available = True
                self.last_vector_time = now
                self._intersection_stable_count = 0
                cte = self._intersection_cte
                turn = (self.intersect_heading_gain * self._intersection_heading
                        + self.intersect_cte_gain * cte)
                self.target_turn = max(TURN_MIN, min(TURN_MAX, turn))
                self.target_speed = self.intersect_speed
                return

            self.vectors_available = False
            return

        # ---- Common tail for non-intersection single-vector and turn-mission cases ----
        self.last_vector_time = now
        raw_error = (lane_center - img_center) / img_center
        self._apply_line_pid(raw_error, now)

        if mission_straight and not self._in_intersection and count == 1:
            self._update_heading_ema(one_vec_heading)

    def _update_lane_width_memory(self, lane_width, img_w):
        if lane_width > 0:
            if self._width_ema_samples == 0:
                self._width_ema = lane_width
            else:
                self._width_ema = 0.15 * lane_width + 0.85 * self._width_ema
            self._width_ema_samples += 1

        if self.learn_lane_width and 150.0 < lane_width < (img_w * 0.65):
            self.learned_lane_width = 0.05 * lane_width + 0.95 * self.learned_lane_width

    def _update_heading_ema(self, heading):
        a_fast = self.intersect_fast_ema_alpha
        a_slow = self.intersect_slow_ema_alpha

        if not self._heading_ema_init:
            self._heading_ema_fast = heading
            self._heading_ema_slow = heading
            self._heading_ema_init = True
        else:
            self._heading_ema_fast = a_fast * heading + (1.0 - a_fast) * self._heading_ema_fast
            self._heading_ema_slow = a_slow * heading + (1.0 - a_slow) * self._heading_ema_slow

    def _apply_line_pid(self, raw_error, now):
        self.error = max(-1.0, min(1.0, raw_error))
        self.target_turn = self._compute_pid(self.error, now)
        self.target_speed = self._compute_speed(self.target_turn)
        self.last_good_turn = self.target_turn

    # ------------------------------------------------------------------
    # PID
    # ------------------------------------------------------------------
    def _compute_pid(self, error, now):
        dt = 0.033 if self.prev_time is None else (now - self.prev_time)
        if dt <= 0.0 or dt > 0.5:
            dt = 0.033

        self.prev_time = now

        p = self.Kp * error
        self.integral += error * dt
        self.integral = max(-0.4, min(0.4, self.integral))
        i = self.Ki * self.integral
        d = self.Kd * (error - self.prev_error) / dt
        self.prev_error = error

        out = p + i + d
        return max(TURN_MIN, min(TURN_MAX, out))

    # ------------------------------------------------------------------
    # Speed shaping
    # ------------------------------------------------------------------
    def _compute_speed(self, turn):
        severity = min(1.0, abs(turn))
        return self.speed_straight - severity * (self.speed_straight - self.speed_sharp)

    def _active_zone_sectors(self):
        """Return LiDAR sector(s) used for safe-zone detection.

        PATIENT: use the ORIGINAL working detector sector exactly as your
        old code did: zone_fov_min_deg..zone_fov_max_deg, default -45°..15°.

        HOSPITAL: keep the same beam-density logic, but look at BOTH side
        sectors because the hospital entrance/wall appears on the side in
        your logs, not in the patient/front sector. Your latest HOSPITAL/LEFT
        log shows close beams near -120°..-90°, while other runs can expose
        the opposite side. Scanning both side sectors fixes hospital without
        touching patient behavior.
        """
        if self._is_hospital_target():
            return [
                (float(self.zone_right_fov_min_deg), float(self.zone_right_fov_max_deg)),
                (float(self.zone_left_fov_min_deg), float(self.zone_left_fov_max_deg)),
            ]

        return [(float(self.zone_fov_min_deg), float(self.zone_fov_max_deg))]

    # ------------------------------------------------------------------
    # LiDAR callback — obstacle avoidance + safe-zone + wall geometry
    # ------------------------------------------------------------------
    def lidar_callback(self, msg):
        ranges = msg.ranges
        n = len(ranges)

        if n == 0 or msg.angle_increment == 0.0:
            return

        # ---- LiDAR orientation map debug ----
        self._last_ranges = ranges
        self._last_range_count = n
        self._last_angle_min = msg.angle_min
        self._last_angle_increment = msg.angle_increment
        self._log_lidar_orientation()

        # ---- Normal obstacle avoidance sector ----
        i_center = int(round((0.0 - msg.angle_min) / msg.angle_increment))

        if self.obstacle_enable:
            half_fov = math.radians(self.obstacle_fov_deg) / 2.0
            i_half = int(round(half_fov / abs(msg.angle_increment)))
            left_min, right_min = self._sector_left_right_min(ranges, n, i_center, i_half)

            self.nearest_dist = min(left_min, right_min)

            if self.nearest_dist < self.obstacle_trigger_dist:
                self.obstacle_detected = True
            elif self.nearest_dist > self.obstacle_clear_dist:
                self.obstacle_detected = False

            if self.obstacle_detected:
                # Turn toward the side with more free space.
                self.obstacle_turn = (self.obstacle_turn_gain
                                      if right_min > left_min else -self.obstacle_turn_gain)
        else:
            self.obstacle_detected = False
            self.obstacle_turn = 0.0
            self.nearest_dist = float('inf')

        # =====================================================================
        # NEW: Parking / safe-zone-entry front obstacle safety sector
        # =====================================================================
        parking_half_fov = math.radians(self.parking_obstacle_fov_deg) / 2.0
        parking_i_half = int(round(parking_half_fov / abs(msg.angle_increment)))
        p_left_min, p_right_min = self._sector_left_right_min(ranges, n, i_center, parking_i_half)

        self.parking_front_left_min = p_left_min
        self.parking_front_right_min = p_right_min
        self.parking_front_min = min(p_left_min, p_right_min)

        if self.parking_front_min < self.parking_obstacle_stop_dist:
            self.parking_obstacle_stop = True
            self.parking_obstacle_detected = True
        elif self.parking_front_min > self.parking_obstacle_clear_dist:
            self.parking_obstacle_stop = False
            self.parking_obstacle_detected = False
        elif self.parking_front_min < self.parking_obstacle_trigger_dist:
            self.parking_obstacle_detected = True

        if self.parking_obstacle_detected:
            # Turn away from closer side.
            # Positive/negative sign may need one-time tuning with steer_sign;
            # this follows the same internal sign convention as normal avoidance.
            self.parking_obstacle_turn = (
                self.parking_obstacle_turn_gain
                if p_right_min > p_left_min else -self.parking_obstacle_turn_gain
            )
        else:
            self.parking_obstacle_turn = 0.0

        # =====================================================================
        # Safe Zone LiDAR sector(s)
        # =====================================================================
        # PATIENT uses the original fixed sector (-45°..15° by default).
        # HOSPITAL uses the same beam-density detector but over both side
        # sectors so the hospital zone is detected before the buggy passes it.
        sectors = self._active_zone_sectors()
        self._zone_active_fov_min_deg = min(a for a, _ in sectors)
        self._zone_active_fov_max_deg = max(b for _, b in sectors)

        valid_distances = []
        sector_data = []
        close_count = 0
        min_dist = float('inf')

        for p_min_deg, p_max_deg in sectors:
            i_p_min = int(round((math.radians(p_min_deg) - msg.angle_min) / msg.angle_increment))
            i_p_max = int(round((math.radians(p_max_deg) - msg.angle_min) / msg.angle_increment))
            i_p_min = max(0, min(n - 1, i_p_min))
            i_p_max = max(0, min(n - 1, i_p_max))

            if i_p_min > i_p_max:
                i_p_min, i_p_max = i_p_max, i_p_min

            for k in range(i_p_min, i_p_max + 1):
                idx = k % n
                r = ranges[idx]
                angle_deg = math.degrees(msg.angle_min + idx * msg.angle_increment)
                sector_data.append((idx, r, angle_deg))

                if math.isfinite(r) and r > 0.05:
                    valid_distances.append(r)
                    min_dist = min(min_dist, r)
                    if self.zone_close_min_dist <= r <= self.zone_close_max_dist:
                        close_count += 1

        self._zone_sector_data = sector_data
        self._zone_valid_distances = valid_distances
        self._zone_total_valid = len(valid_distances)
        self._zone_close_count = close_count
        self._zone_min_dist = min_dist

        # =====================================================================
        # Wall-parallel alignment geometry
        # =====================================================================
        self._update_wall_geometry(ranges, n, msg.angle_min,
                                   msg.angle_increment,
                                   getattr(msg, 'range_max', float('inf')))

    @staticmethod
    def _sector_left_right_min(ranges, n, i_center, i_half):
        left_min = float('inf')
        right_min = float('inf')

        for k in range(i_center - i_half, i_center + i_half + 1):
            r = ranges[k % n]
            if not math.isfinite(r) or r <= 0.05:
                continue
            if k >= i_center:
                left_min = min(left_min, r)
            else:
                right_min = min(right_min, r)

        return left_min, right_min

    def _apply_lane_safety_guard(self, want_turn, want_speed):
        """Strictly prevent obstacle avoidance from driving out of lane.

        The line controller's target_turn is treated as the safe correction
        back toward the lane center. When camera lane error grows, any
        obstacle/parking avoidance turn is blended out and finally replaced
        by the lane correction. This keeps avoidance subordinate to lane
        keeping.
        """
        if not self.lane_guard_enable:
            return want_turn, want_speed

        # Guard only when the camera currently has lane information.
        # Without vectors, we cannot know lane boundaries reliably.
        if not self.vectors_available:
            return want_turn, want_speed

        e = abs(float(self.error))
        correction_turn = max(TURN_MIN, min(TURN_MAX, float(self.target_turn)))

        if e >= self.lane_guard_stop_error:
            # Extremely close to/outside lane edge: almost stop and recover lane.
            return correction_turn, min(want_speed, 0.20)

        if e >= self.lane_guard_hard_error:
            # Hard guard: ignore obstacle turn, steer back into lane.
            return correction_turn, min(want_speed, self.lane_guard_speed_hard)

        if e >= self.lane_guard_soft_error:
            # Soft guard: keep mostly lane correction, allow tiny avoidance.
            # The closer to hard limit, the less obstacle steering remains.
            span = max(1e-6, self.lane_guard_hard_error - self.lane_guard_soft_error)
            t = max(0.0, min(1.0, (e - self.lane_guard_soft_error) / span))
            keep_lane = 0.65 + 0.30 * t
            guarded_turn = keep_lane * correction_turn + (1.0 - keep_lane) * want_turn
            guarded_speed = min(want_speed, self.lane_guard_speed_soft)
            return max(TURN_MIN, min(TURN_MAX, guarded_turn)), guarded_speed

        return want_turn, want_speed

    # ------------------------------------------------------------------
    # Main control loop
    # ------------------------------------------------------------------
    def control_loop(self):
        now = time.time()

        # =================================================================
        # Mission assembly watchdog
        # =================================================================
        accepting = self.mission_state in (
            MissionState.NORMAL_LINE_FOLLOWING,
            MissionState.NAVIGATING_TO_NEXT_TARGET
        )

        if accepting and self.mission_ready:
            self._activate_pending_mission()
        elif (accepting
              and self.pending_target_qr is not None
              and self.pending_target_type is None
              and self.pending_target_qr_time is not None
              and (now - self.pending_target_qr_time) >= self.target_type_wait_timeout):
            self.get_logger().warn(
                f"No /target_type within {self.target_type_wait_timeout:.1f}s of /target_qr — "
                "activating mission with UNKNOWN (legacy) target type."
            )
            self._activate_pending_mission(legacy=True)

        # =================================================================
        # Stopped / idle FSM states
        # =================================================================
        if self.mission_state == MissionState.MISSION_COMPLETE:
            self.publish_drive_cmd(0.0, 0.0)
            return

        if self.mission_state == MissionState.WAITING_FOR_SERVER_ACK:
            self.publish_drive_cmd(0.0, 0.0)
            if now - self._wait_log_time >= 1.0:
                self._wait_log_time = now
                self.get_logger().info(f"[{self.mission_state}] Waiting for Server Assignment...")
            return

        if self.mission_state == MissionState.PARKED_IN_SAFE_ZONE:
            self.publish_drive_cmd(0.0, 0.0)
            return

        # =================================================================
        # ENTERING_SAFE_ZONE — wall-parallel forward-distance parking
        # =================================================================
        if self.mission_state == MissionState.ENTERING_SAFE_ZONE:
            self._update_travel_distance(now)
            self._run_parking_check(now)
            if self.mission_state == MissionState.PARKED_IN_SAFE_ZONE:
                return

        # =================================================================
        # WAITING_FOR_SAFE_ZONE — LiDAR zone detection active
        # =================================================================
        if self.mission_state == MissionState.WAITING_FOR_SAFE_ZONE:
            self._run_safe_zone_detector(now)

        # =================================================================
        # Intersection watchdog
        # =================================================================
        if self._in_intersection:
            if self.last_vector_time is not None and (now - self.last_vector_time) > self.intersect_max_time:
                self._exit_intersection("watchdog")

        # =================================================================
        # Driving logic
        # =================================================================
        if self.mission_state == MissionState.ENTERING_SAFE_ZONE:
            # Parking policy:
            # - ignore line-follow steering
            # - drive straight at parking speed
            # - apply parking obstacle protection
            # - apply wall-parallel alignment
            want_turn = 0.0
            want_speed = self.parking_speed
        else:
            if (self.obstacle_detected
                    and self.mission_state != MissionState.WAITING_FOR_SAFE_ZONE):
                want_turn = max(TURN_MIN, min(TURN_MAX,
                                0.35 * self.target_turn + self.obstacle_turn))
                if self.nearest_dist <= self.obstacle_stop_dist:
                    want_speed = 0.0
                else:
                    want_speed = self.obstacle_speed
            elif self._in_intersection:
                want_turn = self.target_turn
                want_speed = self.target_speed
            elif self.vectors_available:
                want_turn = self.target_turn
                want_speed = self.target_speed
            else:
                elapsed = 1e9 if self.last_vector_time is None else (now - self.last_vector_time)
                if elapsed <= self.no_vector_hold:
                    want_turn = self.last_good_turn
                    want_speed = self.speed_lost
                else:
                    want_turn = self.last_good_turn * 0.5 if self.current_mission == "STRAIGHT" else self.last_good_turn
                    want_speed = self.speed_lost
                self.integral = 0.0
                self.prev_time = None

        # ---- Slow-approach speed cap ----
        if self.mission_state == MissionState.WAITING_FOR_SAFE_ZONE:
            want_speed = min(want_speed, self.slow_approach_speed)

        # ---- PARKING obstacle recovery ----
        # Recovery is ONLY triggered when the buggy is actually too close
        # to an obstacle (parking_obstacle_stop). Normal/medium-range
        # obstacle detections do NOT steer the buggy out of lane.
        if (self.mission_state == MissionState.ENTERING_SAFE_ZONE
                and self.parking_obstacle_enable):

            # Existing recovery maneuver in progress.
            if self._parking_recovery_phase == "REVERSING":
                if now < self._parking_recovery_until:
                    want_speed = float(self.parking_reverse_speed)
                    want_turn = 0.0
                    self.filtered_speed = want_speed
                    self.filtered_turn = 0.0
                else:
                    self._parking_recovery_phase = "ESCAPING"
                    self._parking_recovery_until = now + float(self.parking_escape_time)
                    want_speed = float(self.parking_escape_speed)
                    want_turn = max(TURN_MIN, min(TURN_MAX,
                        self._parking_escape_turn * float(self.parking_escape_turn_gain)))

            elif self._parking_recovery_phase == "ESCAPING":
                if now < self._parking_recovery_until:
                    want_speed = min(want_speed, float(self.parking_escape_speed))
                    want_turn = max(TURN_MIN, min(TURN_MAX,
                        self._parking_escape_turn * float(self.parking_escape_turn_gain)))
                    if self.parking_obstacle_stop:
                        self._parking_recovery_phase = "REVERSING"
                        self._parking_recovery_until = now + float(self.parking_reverse_time)
                        want_speed = float(self.parking_reverse_speed)
                        want_turn = 0.0
                        self.filtered_speed = want_speed
                        self.filtered_turn = 0.0
                else:
                    self._parking_recovery_phase = "NONE"

            # Start recovery ONLY when obstacle is in hard-stop distance.
            if self._parking_recovery_phase == "NONE" and self.parking_obstacle_stop:
                self._parking_recovery_phase = "REVERSING"
                self._parking_recovery_until = now + float(self.parking_reverse_time)
                self._parking_escape_turn = self.parking_obstacle_turn
                want_speed = float(self.parking_reverse_speed)
                want_turn = 0.0
                self.filtered_speed = want_speed
                self.filtered_turn = 0.0
                if self.debug_log and self._tick % 15 == 0:
                    self.get_logger().warn(
                        f"[ParkingObstacle] HIT/TOO CLOSE -> REVERSE front={self.parking_front_min:.2f}m "
                        f"L={self.parking_front_left_min:.2f}m "
                        f"R={self.parking_front_right_min:.2f}m "
                        f"escape_turn={self._parking_escape_turn:+.2f}"
                    )
        # ---- Wall-parallel alignment steering ----
        # Do NOT add wall alignment on top of a hard stop, otherwise it can fight
        # the escape/avoidance turn when the nose is too close to the hospital.
        if (self.mission_state == MissionState.ENTERING_SAFE_ZONE
                and self._wall_align_valid
                and not self.parking_obstacle_stop
                and self._parking_recovery_phase == "NONE"):
            want_turn = max(TURN_MIN, min(TURN_MAX,
                want_turn + self.wall_align_sign * self.wall_align_gain * self._wall_align_error))

        # Final strict lane guard: after obstacle avoidance, parking recovery
        # and wall alignment, lane keeping gets the last word.
        want_turn, want_speed = self._apply_lane_safety_guard(want_turn, want_speed)

        self.filtered_turn = self.steer_alpha * want_turn + (1.0 - self.steer_alpha) * self.filtered_turn
        self.filtered_speed = self.speed_alpha * want_speed + (1.0 - self.speed_alpha) * self.filtered_speed

        final_turn = max(TURN_MIN, min(TURN_MAX, self.filtered_turn))
        min_speed_allowed = -1.0 if self.mission_state == MissionState.ENTERING_SAFE_ZONE else SPEED_MIN
        final_speed = max(min_speed_allowed, min(SPEED_MAX, self.filtered_speed))

        self.publish_drive_cmd(final_speed, self.steer_sign * final_turn)

        self._tick += 1
        if self.debug_log and self._tick % 15 == 0:
            mode = ('PARK_OBS' if (self.mission_state == MissionState.ENTERING_SAFE_ZONE
                                   and self.parking_obstacle_detected) else
                    ('INT' if self._in_intersection else
                     ('OBS' if self.obstacle_detected else 'NORM')))
            self.get_logger().info(
                f"vec={'Y' if self.vectors_available else 'N'} "
                f"side={self.last_single_side} "
                f"width={self.learned_lane_width:.0f} "
                f"mission={self.current_mission} "
                f"target_type={self.active_target_type if self.active_target_type else 'NONE'} "
                f"fsm={self.mission_state} "
                f"mode={mode} "
                f"err={self.error:+.3f} "
                f"obs={'Y' if self.obstacle_detected else 'N'}@{self.nearest_dist:.2f} "
                f"pobs={'Y' if self.parking_obstacle_detected else 'N'}@{self.parking_front_min:.2f} "
                f"turn_int={final_turn:+.3f} "
                f"joy={self.steer_sign * final_turn:+.3f} spd={final_speed:.2f}"
            )

    # =====================================================================
    # Beam-Density Safe Zone Detector
    # =====================================================================
    def _run_safe_zone_detector(self, now):
        if now - self._lidar_debug_time >= 0.5:
            self._lidar_debug_time = now
            self._log_zone_debug(now)

        close_count = self._zone_close_count
        threshold = self.zone_close_beam_threshold

        if close_count >= threshold:
            self._zone_consecutive_scans += 1
            if self._zone_consecutive_scans >= self.zone_confirm_scans:
                self._zone_detection_state = "DETECTED"
            else:
                self._zone_detection_state = "CANDIDATE"

            self.get_logger().info(
                f"[SafeZone] {self._zone_detection_state}  "
                f"close={close_count}/{threshold} beams  "
                f"consecutive={self._zone_consecutive_scans}/{self.zone_confirm_scans}  "
                f"valid={self._zone_total_valid}  min={self._zone_min_dist:.2f}m"
            )

            if self._zone_consecutive_scans >= self.zone_confirm_scans:
                self._on_safe_zone_detected()
        else:
            if self._zone_consecutive_scans > 0:
                self.get_logger().info(
                    f"[SafeZone] MISS — streak reset  close={close_count}/{threshold} beams  "
                    f"(needed {self.zone_confirm_scans} consecutive scans)"
                )
            self._zone_consecutive_scans = 0
            self._zone_detection_state = "MISS"

    # =====================================================================
    # Safe Zone detected — enter the zone
    # =====================================================================
    def _on_safe_zone_detected(self):
        if self._safe_zone_published:
            return

        self._zone_consecutive_scans = 0

        self._wall_side = None
        self._wall_phase = "SEARCHING"
        self._wall_present_count = 0
        self._wall_miss_count = 0
        self._wall_start_dist = 0.0
        self._wall_last_s = None
        self._wall_align_ok = 0
        self._wall_align_error = 0.0
        self._wall_align_valid = False
        self._wall_cross_armed = False
        self._wall_cross_sign = 0
        self._wall_last_align = 0.0
        self._wall_midpoint_dist = 0.0
        self._park_start_time = 0.0
        self._park_start_dist = 0.0
        self._last_travel_at_check = None
        self._travel_dist = 0.0
        self._last_travel_time = None

        self.get_logger().info(
            "========================================\n"
            "SAFE ZONE DETECTED\n"
            f"Target type   : {self.active_target_type}\n"
            f"Park distance : {self._target_parking_distance():.2f} m\n"
            f"Close beams   : {self._zone_close_count} (threshold {self.zone_close_beam_threshold})\n"
            f"Consecutive   : {self.zone_confirm_scans} scans\n"
            f"Total valid   : {self._zone_total_valid} beams\n"
            f"Min distance  : {self._zone_min_dist:.2f}m\n"
            "Entering Safe Zone...\n"
            "========================================"
        )

        self._transition_mission_state(
            MissionState.ENTERING_SAFE_ZONE,
            f"Safe zone detected (close beams {self._zone_close_count}/{self.zone_close_beam_threshold}, "
            f"type={self.active_target_type}, park_dist={self._target_parking_distance():.2f}m)"
        )

    # =====================================================================
    # LiDAR orientation map
    # =====================================================================
    def _log_lidar_orientation(self):
        now = time.time()
        if now - self._orient_map_log_time < 1.0:
            return

        self._orient_map_log_time = now
        ranges = self._last_ranges
        n = self._last_range_count
        if n == 0:
            return

        def fmt(r):
            if not math.isfinite(r):
                return "inf"
            return f"{r:.2f} m"

        lines = ["================ LIDAR ORIENTATION MAP ================"]
        for idx in range(0, 360, 10):
            if idx < n:
                angle_deg = math.degrees(self._last_angle_min + idx * self._last_angle_increment)
                lines.append(f"Beam {idx:3d} | Angle {angle_deg:6.1f}° | {fmt(ranges[idx])}")
            else:
                lines.append(f"Beam {idx:3d} | Angle   N/A | N/A")
        lines.append("=======================================================")
        self.get_logger().info("\n".join(lines))

    # =====================================================================
    # Wall-parallel alignment geometry
    # =====================================================================
    def _update_wall_geometry(self, ranges, n, angle_min, angle_increment, range_max=float('inf')):
        self._scan_seq += 1
        self._wall_geom = {}
        for side in ("LEFT", "RIGHT"):
            self._wall_geom[side] = self._side_wall_geometry(
                ranges, n, angle_min, angle_increment, side, range_max
            )

    def _side_wall_geometry(self, ranges, n, angle_min, angle_increment, side, range_max):
        if side == "RIGHT":
            i0 = int(round((0.0 - angle_min) / angle_increment))
            i1 = int(round((math.pi - angle_min) / angle_increment))
        else:
            i0 = int(round((-math.pi - angle_min) / angle_increment))
            i1 = int(round((0.0 - angle_min) / angle_increment))

        i0 = max(0, min(n - 1, i0))
        i1 = max(0, min(n - 1, i1))

        max_wall = self.wall_max_range_frac * range_max if math.isfinite(range_max) else float('inf')

        runs = []
        cur = []
        gap = 0

        for i in range(i0, i1 + 1):
            r = ranges[i % n]
            solid = (math.isfinite(r) and r > self.wall_min_range_m and r < max_wall)
            if solid:
                cur.append((angle_min + i * angle_increment, r))
                gap = 0
            else:
                if cur:
                    gap += 1
                    if gap > self.wall_max_gap_beams:
                        runs.append(cur)
                        cur = []
                        gap = 0

        if cur:
            runs.append(cur)

        if not runs:
            return None

        main = max(runs, key=len)
        if len(main) < self.wall_min_beams:
            return None

        pts = [(r * math.cos(a), r * math.sin(a)) for a, r in main]

        side_sign = 1.0 if side == "RIGHT" else -1.0
        abeam = 0.5 * math.pi * side_sign
        patch = [p for p in pts
                 if abs(math.atan2(p[1], p[0]) - abeam) <= math.radians(self.wall_fit_patch_deg)]

        if len(patch) < 3:
            patch = pts
        if len(patch) < 3:
            return None

        fit = self._fit_wall_line(patch)
        if fit is None:
            return None

        n_x, n_y, d_line = fit

        u_x, u_y = -n_y, n_x
        if u_x < 0.0:
            u_x, u_y = -u_x, -u_y

        t_vals = []
        for px, py in pts:
            t_vals.append((px - d_line * n_x) * u_x + (py - d_line * n_y) * u_y)

        t_min, t_max = min(t_vals), max(t_vals)
        align_err = math.atan2(u_y, u_x)

        return {
            "s": 0.5 * (t_min + t_max),
            "len_m": t_max - t_min,
            "lateral_m": d_line,
            "align_err_rad": align_err,
            "beams": len(main),
        }

    def _fit_wall_line(self, pts):
        def fit_once(points):
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
            vxx = sum((p[0] - cx) ** 2 for p in points)
            vyy = sum((p[1] - cy) ** 2 for p in points)
            vxy = sum((p[0] - cx) * (p[1] - cy) for p in points)

            tr = vxx + vyy
            det = vxx * vyy - vxy * vxy
            disc = math.sqrt(max(0.0, 0.25 * tr * tr - det))
            lam = 0.5 * tr - disc

            if abs(lam - vxx) > abs(lam - vyy):
                n_x, n_y = 0.0, 1.0
            else:
                n_x, n_y = 1.0, 0.0

            if abs(vxy) > 1e-12 or abs(vxx - lam) > 1e-12:
                n_x, n_y = vxy, lam - vxx

            norm = math.hypot(n_x, n_y)
            if norm < 1e-9:
                return None

            n_x, n_y = n_x / norm, n_y / norm
            if n_x * cx + n_y * cy < 0.0:
                n_x, n_y = -n_x, -n_y

            return n_x, n_y, n_x * cx + n_y * cy

        first = fit_once(pts)
        if first is None:
            return None

        n_x, n_y, d_line = first
        inliers = [p for p in pts
                   if abs(n_x * p[0] + n_y * p[1] - d_line) <= self.wall_fit_tol_m]

        if len(inliers) < 3:
            return None

        return fit_once(inliers)

    # =====================================================================
    # Wall-parallel alignment parking detector
    # =====================================================================
    def _select_wall_side(self):
        g_left = self._wall_geom.get("LEFT")
        g_right = self._wall_geom.get("RIGHT")

        if self.current_mission == "LEFT":
            return "LEFT" if g_left is not None else None
        if self.current_mission == "RIGHT":
            return "RIGHT" if g_right is not None else None

        if g_left is not None and g_right is not None:
            return "LEFT" if g_left["beams"] >= g_right["beams"] else "RIGHT"
        if g_left is not None:
            return "LEFT"
        return "RIGHT" if g_right is not None else None

    def _target_text(self):
        """Combined active target text for robust PATIENT/HOSPITAL matching."""
        return f"{self.active_target_type or ''} {self.active_target_qr or ''}".upper()

    def _is_hospital_target(self):
        return "HOSPITAL" in self._target_text()

    def _is_patient_target(self):
        return "PATIENT" in self._target_text()

    def _target_parking_distance(self):
        """Return parking forward distance.

        Patient and hospital now use the SAME parking distance/logic.
        Keep the separate params declared for live tuning/backward
        compatibility, but do not automatically make hospital drive deeper.
        """
        return float(self.parking_forward_distance)

    def _forward_speed_estimate(self):
        if (self._odom_linear_x is not None and self._odom_time is not None
                and time.time() - self._odom_time < 0.25):
            return max(0.0, self._odom_linear_x)
        return max(0.0, self._last_cmd_speed)

    def _update_travel_distance(self, now):
        if self._last_travel_time is None:
            self._last_travel_time = now
            return

        dt = now - self._last_travel_time
        self._last_travel_time = now

        if dt <= 0.0 or dt > 0.5:
            dt = 0.033

        self._travel_dist += self._forward_speed_estimate() * dt

    def odom_callback(self, msg):
        try:
            self._odom_linear_x = float(msg.twist.twist.linear.x)
            self._odom_time = time.time()
        except Exception:
            pass

    def _run_parking_check(self, now):
        if self._scan_seq == self._park_check_seq:
            return

        self._park_check_seq = self._scan_seq

        if self._park_start_time == 0.0:
            self._park_start_time = now
            self._park_start_dist = self._travel_dist
            self._wall_phase = "DRIVING"

        # ---- Wall side / alignment bookkeeping ----
        side = self._select_wall_side()
        self._wall_side = side
        g = self._wall_geom.get(side) if side else None

        if g is not None:
            self._wall_align_error = g["align_err_rad"]
            self._wall_align_valid = True
            self._wall_len_m = g["len_m"]
            self._wall_lateral_m = g["lateral_m"]
            self._wall_last_align = self._wall_align_error
            self._wall_miss_count = 0
        else:
            self._wall_align_valid = False
            self._wall_miss_count += 1

        # ---- Distance-based parking stop ----
        driven = self._travel_dist - self._park_start_dist
        target_dist = self._target_parking_distance()

        if driven >= target_dist:
            self._on_parking_complete()
            return

        if now - self._park_start_time > self.parking_timeout_s:
            if self.mission_state == MissionState.ENTERING_SAFE_ZONE:
                self.get_logger().warn(
                    f"Parking watchdog ({self.parking_timeout_s:.0f}s) — parking at current position."
                )
            self._on_parking_complete()
            return

        self._log_parking(now)

    def _log_parking(self, now):
        if now - self._parking_log_time < 0.5:
            return

        self._parking_log_time = now
        side = self._wall_side
        g = self._wall_geom.get(side) if side else None
        present = g is not None
        align_deg = math.degrees(self._wall_align_error) if self._wall_align_valid else 0.0
        align_txt = f"{align_deg:+.1f} deg" if self._wall_align_valid else "n/a"
        driven = self._travel_dist - self._park_start_dist

        self.get_logger().info(
            "========== PARKING (FORWARD DISTANCE) ==========\n"
            f"Target type        : {self.active_target_type}\n"
            f"Side               : {side}\n"
            f"Wall present       : {'YES' if present else 'NO'}\n"
            f"Lateral distance   : {self._wall_lateral_m:.2f} m\n"
            f"Align error        : {align_txt}\n"
            f"Forward distance   : {driven:.2f} / {self._target_parking_distance():.2f} m\n"
            f"Parking obstacle   : {'STOP' if self.parking_obstacle_stop else ('AVOID' if self.parking_obstacle_detected else 'CLEAR')}\n"
            f"Recovery phase     : {self._parking_recovery_phase}\n"
            f"Front min          : {self.parking_front_min:.2f} m\n"
            "================================================="
        )

    # =====================================================================
    # Parking complete — stop, publish /safe_zone, wait for QR Detector
    # =====================================================================
    def _on_parking_complete(self):
        if self._safe_zone_published:
            return

        self._safe_zone_published = True

        self.publish_drive_cmd(0.0, 0.0)

        zone_msg = Bool()
        zone_msg.data = True
        self.pub_safe_zone.publish(zone_msg)

        self.get_logger().info(
            "========================================\n"
            "SAFE ZONE FULLY ENTERED\n"
            f"Target type : {self.active_target_type}\n"
            f"Distance    : {self._target_parking_distance():.2f} m target\n"
            "Stopping buggy...\n"
            "Publishing /safe_zone\n"
            "Waiting for QR Detector...\n"
            "========================================"
        )

        self._transition_mission_state(
            MissionState.PARKED_IN_SAFE_ZONE,
            "parking target distance covered — /safe_zone published"
        )

    # =====================================================================
    # LiDAR debug logging for safe-zone sector
    # =====================================================================
    def _log_zone_debug(self, now):
        sector = self._zone_sector_data
        if not sector:
            return

        total_count = len(sector)
        valid_count = self._zone_total_valid
        close_count = self._zone_close_count
        threshold = self.zone_close_beam_threshold
        min_dist = self._zone_min_dist

        def fmt(r):
            if not math.isfinite(r):
                return 'inf'
            return f'{r:.2f}'

        self.get_logger().info(
            "================ LIDAR DEBUG ================\n"
            f"Sector       : {self._zone_active_fov_min_deg:.0f}° to {self._zone_active_fov_max_deg:.0f}°\n"
            f"Total beams  : {total_count}\n"
            f"Valid beams  : {valid_count}\n"
            f"Close beams  : {close_count}/{threshold}  "
            f"(band {self.zone_close_min_dist:.2f}–{self.zone_close_max_dist:.2f} m)\n"
            f"Min distance : {fmt(min_dist)}\n"
            f"Detection    : {self._zone_detection_state}  "
            f"(consecutive {self._zone_consecutive_scans}/{self.zone_confirm_scans})\n"
            f"Target type  : {self.active_target_type if self.active_target_type else 'NONE'}\n"
            f"Park dist    : {self._target_parking_distance():.2f} m\n"
            f"FSM state    : {self.mission_state}\n"
            "============================================"
        )

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------
    def publish_drive_cmd(self, speed, turn):
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        msg.axes = [0.0, float(speed), 0.0, float(turn)]
        self._last_cmd_speed = float(speed)
        self.pub_joy.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
