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
#
# States and their meaning:
#
#   NORMAL_LINE_FOLLOWING    — Default.  Pure line following, no zone
#                              detection.  Entered on startup and after
#                              a new /mission/turn restarts the cycle.
#
#   WAITING_FOR_SAFE_ZONE    — /target_qr received.  The buggy continues
#                              line following at SLOW APPROACH speed
#                              (slow_approach_speed) while the LiDAR
#                              safe-zone detector is active.  The only
#                              way OUT of this state is a confirmed zone
#                              detection (→ ENTERING_SAFE_ZONE).
#
#   ENTERING_SAFE_ZONE       — Zone detected.  The buggy continues line
#                              following at PARKING speed (parking_speed)
#                              while the wall-parallel ALIGNMENT detector
#                              continuously reconstructs the mission-side
#                              wall (line fit) and steers parallel to it.
#                              The buggy drives STRAIGHT into the parking
#                              area for parking_forward_distance meters
#                              (measured with odometry only), maintaining
#                              wall-parallel alignment the whole way.  It
#                              stops when that forward distance is reached
#                              (→ PARKED_IN_SAFE_ZONE).  NO wall-midpoint
#                              / wall-length / segment-midpoint / s-crossing
#                              logic is used anywhere in the stop decision.
#
#   PARKED_IN_SAFE_ZONE      — Robot stopped at its final parking
#                              position.  /safe_zone is published HERE —
#                              the ONLY place in the code.  The QR
#                              Detector receives it, sends the target QR
#                              to the server, and later publishes
#                              /resume_line_following "RESUME"
#                              (→ NAVIGATING_TO_NEXT_TARGET) or
#                              "MISSION_COMPLETE" (→ MISSION_COMPLETE).
#
#   WAITING_FOR_SERVER_ACK   — (legacy) stopped, waiting for the QR
#                              Detector / server.  Kept for
#                              compatibility; the current flow parks
#                              into PARKED_IN_SAFE_ZONE instead.
#
#   NAVIGATING_TO_NEXT_TARGET — Server assigned the next target; the
#                              buggy resumes line following.  The next
#                              /target_qr will push it back into
#                              WAITING_FOR_SAFE_ZONE.
#
#   MISSION_COMPLETE          — All deliveries done.  Stopped, fully
#                              silent idle (one "Waiting For New Goal
#                              Assignment..." banner on entry, then no
#                              logs, timers or polling).  The buggy
#                              ignores all navigation inputs until the QR
#                              Detector publishes a new mission available
#                              notification (/mission/available), which
#                              immediately starts the next cycle
#                              (→ NAVIGATING_TO_NEXT_TARGET).
#
# =====================================================================

# =====================================================================
# PARKING THEORY (ENTERING_SAFE_ZONE)
# =====================================================================
#
# The old "universal wall-midpoint" parking detector (which reconstructed
# the wall, computed its segment midpoint s, and stopped when s crossed
# zero) has been REPLACED with a simple, robust forward-distance drive:
#
#   * The mission-side wall is still reconstructed as a local line on
#     every scan (robust two-pass fit on the abeam patch) — but ONLY to
#     feed the wall-vs-heading ALIGNMENT steering term, so the buggy
#     stops parallel to the building.
#
#   * The buggy LOCKs its heading on entering the zone and drives
#     STRAIGHT into the parking area.  Line-following steering is ignored
#     while parking; the only lateral correction is the wall-parallel
#     alignment term.
#
#   * The forward distance is measured with odometry only (the integrated
#     travelled distance).  When travelled_distance reaches the
#     configurable parking_forward_distance (default ~1.2 m), the robot
#     stops, publishes /safe_zone exactly once, and parks.
#
#   * Recovery: a time-based watchdog never lets the buggy drive forever.
#     Brief wall loss simply disables the alignment term (drive straight);
#     longer absence is safe because the stop is distance-based, not
#     wall-based.
#
class MissionState:

    NORMAL_LINE_FOLLOWING    = "NORMAL_LINE_FOLLOWING"

    WAITING_FOR_SAFE_ZONE    = "WAITING_FOR_SAFE_ZONE"

    ENTERING_SAFE_ZONE       = "ENTERING_SAFE_ZONE"

    PARKED_IN_SAFE_ZONE      = "PARKED_IN_SAFE_ZONE"

    WAITING_FOR_SERVER_ACK   = "WAITING_FOR_SERVER_ACK"

    NAVIGATING_TO_NEXT_TARGET = "NAVIGATING_TO_NEXT_TARGET"

    MISSION_COMPLETE         = "MISSION_COMPLETE"

class LineFollower(Node):

    def __init__(self):

        super().__init__('line_follower')

        # ---------------- Parameters (all live-tunable via ros2 param set) --------
        self.declare_parameter('steer_sign', -1.0)

        self.declare_parameter('Kp', 0.55)

        self.declare_parameter('Ki', 0.0)

        self.declare_parameter('Kd', 0.18)

        self.declare_parameter('lookahead_blend', 0.6)

        self.declare_parameter('lane_width_px', 240.0)

        self.declare_parameter('learn_lane_width', True)

        self.declare_parameter('single_vector_side_margin', 0.20)

        self.declare_parameter('speed_straight', 0.65)

        self.declare_parameter('speed_sharp', 0.35)

        self.declare_parameter('speed_lost', 0.30)

        self.declare_parameter('steer_alpha', 0.55)

        self.declare_parameter('speed_alpha', 0.15)

        self.declare_parameter('no_vector_hold', 0.6)

        self.declare_parameter('debug_log', True)

        self.declare_parameter('obstacle_enable', True)

        self.declare_parameter('obstacle_trigger_dist', 0.90)

        self.declare_parameter('obstacle_clear_dist', 1.30)

        self.declare_parameter('obstacle_fov_deg', 60.0)

        self.declare_parameter('obstacle_turn_gain', 0.9)

        self.declare_parameter('obstacle_speed', 0.32)

        # --- edge-safety margin so the aim point never sits right on the curb ---
        self.declare_parameter('turn_edge_margin_px', 35.0)

        # --- junction lane-width ratio that triggers "wide crossing" handling ---
        self.declare_parameter('junction_width_ratio', 1.8)

        # Straight-intersection state-machine parameters
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

        # =====================================================================
        # Slow approach + wall-parallel parking parameters (noise/control only)
        # =====================================================================
        self.declare_parameter('slow_approach_speed', 0.28)

        self.declare_parameter('parking_speed', 0.20)

        # --- Forward distance to drive into the parking area (odometry only) ---
        self.declare_parameter('parking_forward_distance', 1.2)

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

        self.declare_parameter('parking_timeout_s', 15.0)

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

        self.target_qr_string = ""          # stored for debug logging only

        self.last_valid_mission = "NONE"

        # Beam-density safe-zone detection bookkeeping
        # (used only in WAITING_FOR_SAFE_ZONE)
        self._zone_close_count = 0          # close beams in the latest scan

        self._zone_total_valid = 0          # valid beams in the latest scan

        self._zone_min_dist = float('inf')  # min valid distance, latest scan

        self._zone_consecutive_scans = 0    # consecutive positive scans

        self._zone_detection_state = "MISS"  # MISS / CANDIDATE / DETECTED

        self._safe_zone_published = False   # True once /safe_zone has been
                                            # published for this target —
                                            # prevents duplicate publications

        # Parking bookkeeping (wall-parallel alignment detector,
        # ENTERING_SAFE_ZONE only)
        self._wall_geom = {}                # per-side wall geometry (scan)

        self._wall_side = None              # "LEFT" / "RIGHT" in use

        self._wall_phase = "SEARCHING"      # SEARCHING / DRIVING

        self._wall_present_count = 0        # consecutive wall-present scans

        self._wall_miss_count = 0           # consecutive missing-wall scans

        self._wall_start_dist = 0.0         # travelled dist at wall start

        self._wall_last_s = None            # (legacy, kept for compat)

        self._wall_align_ok = 0             # (legacy, kept for compat)

        self._wall_align_error = 0.0        # wall-vs-heading angle (rad)

        self._wall_align_valid = False      # True when the fit is fresh

        self._wall_cross_armed = False      # (legacy, kept for compat)

        self._wall_cross_sign = 0           # (legacy, kept for compat)

        self._wall_last_align = 0.0         # last known alignment (recovery)

        self._wall_len_m = 0.0              # measured wall length (debug)

        self._wall_lateral_m = 0.0          # lateral distance (debug)

        self._wall_midpoint_dist = 0.0      # (legacy, kept for compat)

        self._park_start_time = 0.0         # watchdog base time

        self._park_start_dist = 0.0         # travelled dist at park start

        self._last_travel_at_check = None   # travel at previous scan check

        self._parking_log_time = 0.0        # debug throttle (0.5 s)

        self._scan_seq = 0                  # scan counter (lidar_callback)

        self._park_check_seq = -1           # last scan processed by parking

        # Travelled distance (odometry twist if available, else commanded
        # speed dead-reckoning) — used ONLY for the parking forward-distance
        # stop and recovery, never for any other control decision.
        self._travel_dist = 0.0

        self._last_travel_time = None

        self._last_cmd_speed = 0.0

        self._odom_linear_x = None

        self._odom_time = None

        # LiDAR orientation-map debug (printed once per second — the
        # beams 0, 10, 20, ..., 350 of the latest scan with their true
        # angles, so the FRONT / LEFT / RIGHT / FRONT-LEFT /
        # FRONT-RIGHT indices can be identified)
        self._orient_map_log_time = 0.0

        self._last_ranges = []

        self._last_range_count = 0

        self._last_angle_min = 0.0

        self._last_angle_increment = 1.0

        # =====================================================================
        # Mission synchronization state (atomic target_type + target_qr)
        # =====================================================================
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

        # LiDAR zone analysis results (written by lidar_callback, read by
        # control_loop).  All inf/NaN are filtered out before these are
        # computed.
        self._zone_valid_distances = []     # list of valid distances this frame

        self._zone_sector_data = []         # raw sector data for debug logging

        # Timers for periodic log messages
        self._wait_log_time = 0.0

        self._lidar_debug_time = 0.0

        # ---------------- ROS plumbing ----------------
        self.create_subscription(
            EdgeVectors, '/edge_vectors', self.edge_vectors_callback, QOS_PROFILE_DEFAULT)

        self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, QOS_PROFILE_DEFAULT)

        self.create_subscription(
            String, '/mission/turn', self.mission_callback, QOS_PROFILE_DEFAULT)

        self.create_subscription(
            String, '/target_qr', self.target_qr_callback, QOS_PROFILE_DEFAULT)

        self.create_subscription(
            String, '/target_type', self.target_type_callback, QOS_PROFILE_DEFAULT)

        self.create_subscription(
            String, '/resume_line_following', self.resume_callback, 10)

        self.create_subscription(
            String, '/mission/available', self.mission_available_callback, 10)

        self.create_subscription(
            Odometry, '/odom', self.odom_callback, QOS_PROFILE_DEFAULT)

        self.pub_joy = self.create_publisher(Joy, '/cerebri/in/joy', QOS_PROFILE_DEFAULT)

        self.pub_safe_zone = self.create_publisher(Bool, '/safe_zone', QOS_PROFILE_DEFAULT)

        self.create_timer(0.033, self.control_loop)

        self.create_timer(1.0, self._reload_params)

        self.get_logger().info(
            "Lane-following controller loaded.\n"
            f"Mission FSM initial state: {self.mission_state}")

    def _transition_mission_state(self, new_state, reason=""):
        """Log and execute a mission state transition."""
        old = self.mission_state
        if old == new_state:
            return
        self.mission_state = new_state

        # When the FSM returns to an idle/accepting state, promote any
        # assignment that was queued while a mission was in progress.
        if new_state in (MissionState.NORMAL_LINE_FOLLOWING,
                         MissionState.NAVIGATING_TO_NEXT_TARGET):
            self._promote_pending_next()

        if self.mission_state != new_state:
            # Promotion already triggered a further transition (the
            # pending mission was activated immediately).
            self.get_logger().info(
                f"MISSION STATE: {old} → {new_state} → {self.mission_state}"
                f"  ({reason})" if reason else "")
        else:
            self.get_logger().info(
                f"MISSION STATE: {old} → {new_state}"
                f"  ({reason})" if reason else "")

    # =====================================================================
    # Safe-zone detection reset
    # =====================================================================
    def _reset_zone_detection(self):
        """Clear all safe-zone detection counters so a fresh cycle can
        begin when the next mission activates.

        NOTE: this ONLY resets the LiDAR beam-density and parking
        bookkeeping.  The active mission context (active_target_type /
        active_target_qr) and the pending/queued mission assembly state
        are intentionally NOT touched here.
        """
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

    # ------------------------------------------------------------------
    # Parameter reload (called once on init and every 1s thereafter)
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
        self.turn_edge_margin_px = g('turn_edge_margin_px')
        self.junction_width_ratio = g('junction_width_ratio')

        # Intersection parameters
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

        # Safe Zone parameters (beam-density approach)
        self.zone_close_min_dist = g('zone_close_min_dist')
        self.zone_close_max_dist = g('zone_close_max_dist')
        self.zone_close_beam_threshold = int(g('zone_close_beam_threshold'))
        self.zone_confirm_scans = int(g('zone_confirm_scans'))
        self.zone_fov_min_deg = g('zone_fov_min_deg')
        self.zone_fov_max_deg = g('zone_fov_max_deg')

        # Slow approach + wall-parallel parking parameters
        self.slow_approach_speed = g('slow_approach_speed')
        self.parking_speed = g('parking_speed')
        self.parking_forward_distance = g('parking_forward_distance')
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

        # Mission synchronization
        self.target_type_wait_timeout = g('target_type_wait_timeout')

    # ------------------------------------------------------------------
    # Target Type callback (mission synchronization)
    # ------------------------------------------------------------------
    def target_type_callback(self, msg):
        """Receive /target_type from the QR Detector ("PATIENT"/"HOSPITAL").

        /target_type is the AUTHORITATIVE mission context.  It is never
        applied directly to the FSM: it is stored as the pending target
        type, and the FSM only transitions to WAITING_FOR_SAFE_ZONE once
        BOTH /target_type and /target_qr for the same assignment have
        arrived (see _activate_pending_mission).  This makes mission
        updates atomic and immune to cross-topic ordering races.

        - Accepting states (NORMAL_LINE_FOLLOWING / NAVIGATING_TO_NEXT_TARGET):
          assemble the incoming mission (type + QR) and activate when
          complete.

        - Busy states (WAITING_FOR_SAFE_ZONE / ENTERING_SAFE_ZONE /
          PARKED_IN_SAFE_ZONE / WAITING_FOR_SERVER_ACK / MISSION_COMPLETE):
          the active mission is IMMUTABLE — the new assignment is queued
          as pending_next_* and promoted only after the current mission
          finishes.
        """
        if not msg.data or not msg.data.strip():
            return
        value = msg.data.strip().upper()
        if value not in ("PATIENT", "HOSPITAL"):
            self.get_logger().info(
                f"Ignoring unknown /target_type value: {msg.data}")
            return

        if self._mission_in_progress():
            # ---- Busy: queue the assignment, never touch the active mission ----
            if self.pending_next_qr is None and self.pending_next_type is None:
                self.pending_next_type = value
                self.get_logger().info(
                    f"New assignment queued while {self.mission_state}: "
                    f"target type {value} (pending).")
            elif self.pending_next_type == value:
                self.get_logger().info(
                    f"Duplicate /target_type ignored: {value}")
            else:
                self.pending_next_type = value
                self.get_logger().warn(
                    f"Pending target type replaced with {value} "
                    f"(current mission still active).")
            return

        # ---- Accepting states: assemble the mission atomically ----
        if self.pending_target_qr is not None and self.pending_target_type is None:
            # The QR arrived first (cross-topic race) — this type completes it.
            self.pending_target_type = value
            self.get_logger().info(
                f"Target type received: {value} — matches pending QR "
                f"{self.pending_target_qr}; mission ready.")
            self.mission_ready = True
            self._activate_pending_mission()
            return

        if self.pending_target_type == value:
            self.get_logger().info(f"Duplicate /target_type ignored: {value}")
            return

        if self.pending_target_type is None:
            self.pending_target_type = value
            self.get_logger().info(
                f"Target type received: {value} — "
                "awaiting /target_qr to activate mission.")
        else:
            # A different type arrived while waiting for its QR — the old
            # pending type was orphaned (its QR never came).  Replace it.
            self.pending_target_type = value
            self.get_logger().warn(
                f"Orphaned pending target type replaced with {value}.")

    # ------------------------------------------------------------------
    # Target QR callback (mission synchronization)
    # ------------------------------------------------------------------
    def target_qr_callback(self, msg):
        """Receive target QR availability from the QR Detector.

        Together with /target_type this forms the atomic mission update:
        the FSM transitions to WAITING_FOR_SAFE_ZONE only when BOTH have
        arrived for the same assignment (see _activate_pending_mission).

        A QR that arrives before its /target_type (cross-topic race) is
        held pending — the FSM does NOT move until the type arrives (or
        the legacy timeout falls back to UNKNOWN).

        Duplicate re-transmissions of the same (type, QR) are ignored —
        they never restart or reset the FSM or the beam-density streak.

        Assignments that arrive while a mission is in progress are
        queued (pending_next_*) and promoted only after the current
        mission completes.

        Backward compatibility: with an older QR Detector that does not
        publish /target_type, the QR is kept pending and the mission is
        activated with "UNKNOWN" after target_type_wait_timeout (legacy
        parking profile).
        """
        if not msg.data or not msg.data.strip():
            return
        qr = msg.data.strip()

        if self._mission_in_progress():
            # ---- Busy: never touch the active mission ----
            if qr == self.active_target_qr:
                self.get_logger().info(
                    f"Duplicate /target_qr ignored (active mission: {qr}).")
                # Clean up a matching type that may have been queued
                # before this QR confirmed the pair as a duplicate.
                if (self.pending_next_qr is None
                        and self.pending_next_type == self.active_target_type):
                    self.pending_next_type = None
                return
            if self.pending_next_qr == qr:
                self.get_logger().info(f"Duplicate queued QR ignored: {qr}")
                return
            self.pending_next_qr = qr
            self.pending_next_qr_time = time.time()
            self.get_logger().info(
                f"New assignment queued while {self.mission_state}: "
                f"QR {qr} (pending).")
            return

        # ---- Accepting states: assemble the mission atomically ----
        if self.pending_target_type is not None and self.pending_target_qr is None:
            # The type already arrived — this QR completes the mission.
            self.pending_target_qr = qr
            self.pending_target_qr_time = time.time()
            self.get_logger().info(
                f"Target QR received: {qr} — matches pending type "
                f"{self.pending_target_type}; mission ready.")
            self.mission_ready = True
            self._activate_pending_mission()
            return

        if self.pending_target_qr == qr:
            self.get_logger().info(f"Duplicate /target_qr ignored: {qr}")
            return

        if self.pending_target_qr is None:
            # QR arrived before its /target_type (cross-topic race), or a
            # legacy QR Detector that never publishes /target_type.
            self.pending_target_qr = qr
            self.pending_target_qr_time = time.time()
            self.get_logger().info(
                f"Target QR received before /target_type: {qr} — "
                f"awaiting /target_type (legacy fallback after "
                f"{self.target_type_wait_timeout:.1f}s).")
        else:
            # A different QR while waiting for its type — replace it.
            self.pending_target_qr = qr
            self.pending_target_qr_time = time.time()
            self.get_logger().warn(
                f"Pending QR replaced with {qr} (awaiting its /target_type).")

    # =====================================================================
    # Mission synchronization helpers
    # =====================================================================
    def _mission_in_progress(self):
        """True while a mission is active/busy, so new assignments must
        be queued instead of applied."""
        return self.mission_state in (
            MissionState.WAITING_FOR_SAFE_ZONE,
            MissionState.ENTERING_SAFE_ZONE,
            MissionState.PARKED_IN_SAFE_ZONE,
            MissionState.WAITING_FOR_SERVER_ACK,
            MissionState.MISSION_COMPLETE,
        )

    def _clear_pending_assembly(self):
        """Reset the incoming-assignment assembly slots."""
        self.pending_target_type = None
        self.pending_target_qr = None
        self.pending_target_qr_time = None
        self.mission_ready = False

    def _activate_pending_mission(self, legacy=False):
        """Atomically activate a fully assembled mission.

        Called only when BOTH the target type and the target QR are
        available (or the legacy timeout fired).  Snapshots the mission
        context into the immutable active_* fields and transitions the
        FSM to WAITING_FOR_SAFE_ZONE.

        Duplicate (type, QR) pairs from re-transmissions are ignored —
        the FSM is never restarted for them.
        """
        ttype = "UNKNOWN" if legacy else self.pending_target_type
        qr = self.pending_target_qr
        if qr is None:
            # Nothing to activate yet (defensive).
            self._clear_pending_assembly()
            return
        if ttype is None:
            ttype = "UNKNOWN"

        # ---- Duplicate re-transmission guard ----
        key = (ttype, qr)
        if key == self._last_mission_key:
            self.get_logger().info(
                f"Duplicate mission ignored (already processed): "
                f"type={ttype}, QR={qr}")
            self._clear_pending_assembly()
            return
        self._last_mission_key = key

        # ---- Cache the mission context (immutable snapshot) ----
        self.active_target_type = ttype
        self.active_target_qr = qr
        self.target_qr_string = qr
        self.get_logger().info(
            "Mission cached:\n"
            f"Target Type : {ttype}\n"
            f"QR          : {qr}")

        self._clear_pending_assembly()
        self._reset_zone_detection()
        self._transition_mission_state(
            MissionState.WAITING_FOR_SAFE_ZONE,
            f"/target_qr received: {qr} (target type: {ttype})")
        self.get_logger().info("Mission activated.")

    def _complete_active_mission(self, reason=""):
        """Release the cached mission context after it has finished.

        The active mission is immutable until this point; only after it
        completes can a queued assignment be promoted.  The duplicate-
        mission key is cleared here so that in continuous missions the
        SAME target (e.g. PATIENT_1 again) can be legitimately
        re-assigned in a later cycle.
        """
        if self.active_target_type is not None:
            self.get_logger().info(
                "Mission completed.\n"
                f"Reason        : {reason}\n"
                f"Target Type   : {self.active_target_type}\n"
                f"QR            : {self.active_target_qr}")
        self.active_target_type = None
        self.active_target_qr = ""
        self.target_qr_string = ""
        self._last_mission_key = None
        self._reset_zone_detection()

    def _promote_pending_next(self):
        """Promote a queued assignment once the FSM is idle again.

        Called on every transition into NORMAL_LINE_FOLLOWING or
        NAVIGATING_TO_NEXT_TARGET.  If both pieces of the queued
        assignment are present it is activated immediately; otherwise
        the missing piece is awaited (with the legacy timeout fallback).
        """
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
        """Receive /resume_line_following from QR Detector.

        Two valid payloads:
          "RESUME"          — server assigned the next target; resume
                               line following.
          "MISSION_COMPLETE" — all deliveries done; enter MISSION_COMPLETE.

        Valid from the stopped wait states: PARKED_IN_SAFE_ZONE (current
        flow) and WAITING_FOR_SERVER_ACK (legacy).
        """
        received = msg.data.strip() if msg.data else ""
        if received not in ("RESUME", "MISSION_COMPLETE"):
            self.get_logger().info(f"Ignoring resume message: {received}")
            return
        self._process_resume(received)

    def _process_resume(self, received):
        """Apply a RESUME / MISSION_COMPLETE message."""
        if self.mission_state not in (
                MissionState.PARKED_IN_SAFE_ZONE,
                MissionState.WAITING_FOR_SERVER_ACK):
            self.get_logger().info(
                f"{received} ignored in state {self.mission_state}")
            return

        if received == "MISSION_COMPLETE":
            self._complete_active_mission("MISSION_COMPLETE received")
            self._transition_mission_state(
                MissionState.MISSION_COMPLETE,
                "Server signalled mission complete")
            # One-time banner when entering the silent idle state.  After
            # this, NO periodic logs, NO timers, NO polling — the buggy
            # stays stopped until /mission/available wakes it.
            self.get_logger().info(
                "====================================\n"
                "Waiting For New Goal Assignment...\n"
                "====================================")
            return

        # RESUME
        self._complete_active_mission("RESUME received")
        self._transition_mission_state(
            MissionState.NAVIGATING_TO_NEXT_TARGET,
            "Server ACK / RESUME received")
        self._wait_log_time = 0.0

    # ------------------------------------------------------------------
    # Mission topic callback
    # ------------------------------------------------------------------
    def mission_callback(self, msg):
        mission = msg.data
        # --- Handle NONE / empty / whitespace ---
        if not mission or not mission.strip() or mission.strip() == "NONE":
            self.get_logger().info("Mission Topic Received: NONE — ignoring")
            return
        mission = mission.strip()
        if mission not in ["LEFT", "RIGHT", "STRAIGHT"]:
            return
        self.get_logger().info(f"Mission Topic Received: {mission}")

        # --- MISSION_COMPLETE: navigation is IGNORED until a new mission
        #     is received via /mission/available.  /mission/turn belongs
        #     to the object recognition node and only steers DURING
        #     navigation; it must never restart the buggy.
        if self.mission_state == MissionState.MISSION_COMPLETE:
            self.get_logger().info(
                f"Mission Topic ignored in MISSION_COMPLETE: {mission} "
                "(waiting for /mission/available)")
            return

        # --- WAITING_FOR_SERVER_ACK: accept new mission direction ---
        if self.mission_state == MissionState.WAITING_FOR_SERVER_ACK:
            if mission == self.last_valid_mission:
                self.get_logger().info("Ignoring duplicate mission")
                return
            self.get_logger().info(
                "Assignment Accepted — Resuming Navigation")
            self._reset_zone_detection()
            self._complete_active_mission("New mission direction")
            self._transition_mission_state(
                MissionState.NAVIGATING_TO_NEXT_TARGET,
                "New mission direction while waiting for server")
            self.last_valid_mission = mission
            self.current_mission = mission
            self._straight_junction_side = None
            self._reset_intersection_state()
            self._heading_ema_init = False
            return

        # --- WAITING_FOR_SAFE_ZONE: new mission may change direction ---
        if self.mission_state == MissionState.WAITING_FOR_SAFE_ZONE:
            if mission != self.current_mission:
                self.get_logger().info(
                    f"Mission changed to {mission} while WAITING_FOR_SAFE_ZONE")
                self.current_mission = mission
                self._straight_junction_side = None
                self._reset_intersection_state()
                self._heading_ema_init = False
            self.last_valid_mission = mission
            return

        # --- Normal (line-following) mission update ---
        if mission != self.current_mission:
            self.get_logger().info(f"Mission changed to {mission}")
            self.current_mission = mission
            self._straight_junction_side = None
            self._reset_intersection_state()
            self._heading_ema_init = False
        self.last_valid_mission = mission

    # ------------------------------------------------------------------
    # New Mission Available callback (continuous missions)
    # ------------------------------------------------------------------
    def mission_available_callback(self, msg):
        """Receive /mission/available from the QR Detector.

        The QR Detector publishes this the moment the Municipality Server
        assigns the NEXT target (e.g. the next patient after a hospital
        mission), with the target identifier as the payload.

        The Line Follower validates the payload (must be a PATIENT_x /
        HOSPITAL_x target identifier) and immediately transitions:

            MISSION_COMPLETE
                ↓
            NAVIGATING_TO_NEXT_TARGET

        Line following resumes immediately — /mission/turn is NOT
        required to start moving.  The object recognition node remains
        the only publisher of /mission/turn; it only updates the turn
        direction later while navigating.  No additional resume message
        is needed.  Event-driven: no polling, no timers.
        """
        if not msg.data or not msg.data.strip():
            return
        assignment = msg.data.strip()

        # --- Verify that a valid target identifier was received ---
        upper = assignment.upper()
        if not (upper.startswith("PATIENT_") or upper.startswith("HOSPITAL_")):
            self.get_logger().info(
                f"/mission/available ignored: invalid target payload: "
                f"{assignment}")
            return

        if self.mission_state in (
                MissionState.MISSION_COMPLETE,
                MissionState.PARKED_IN_SAFE_ZONE,
                MissionState.WAITING_FOR_SERVER_ACK):
            # Normal path (MISSION_COMPLETE: hospital mission finished,
            # next target assigned) or a race where the server's
            # next-target assignment arrived before the MISSION_COMPLETE
            # message was processed (PARKED_IN_SAFE_ZONE /
            # WAITING_FOR_SERVER_ACK).  Either way the previous mission
            # is over — start navigating immediately.
            self._complete_active_mission("New mission assigned by QR Detector")
            self._transition_mission_state(
                MissionState.NAVIGATING_TO_NEXT_TARGET,
                f"New mission assigned by QR Detector ({assignment})")
            return

        self.get_logger().info(
            f"/mission/available ignored in state {self.mission_state}")

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
        """Keep the aim point at least turn_edge_margin_px away from either edge."""
        margin = self.turn_edge_margin_px
        if lane_width <= 2.0 * margin:
            return lane_width / 2.0
        return max(margin, min(lane_width - margin, offset))

    # ------------------------------------------------------------------
    # Intersection state-machine helpers
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
        if p1.y >= p0.y:
            near, far = p1, p0
        else:
            near, far = p0, p1
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
                f"clamping to 0 (straight). reason={reason}")
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
            f"{' (clamped)' if clamped_hard else ''}")

    def _exit_intersection(self, reason):
        self.get_logger().info(
            f"*** Exiting STRAIGHT_INTERSECTION reason={reason} "
            f"stable={self._intersection_stable_count}")
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
                    width_ok = (lane_width <=
                                self.learned_lane_width * self.intersect_exit_width_ratio
                                and lane_width > 0)
                    if width_ok:
                        self._intersection_stable_count += 1
                        blend = self.intersect_heading_blend
                        self._intersection_heading = (
                            (1.0 - blend) * self._intersection_heading
                            + blend * current_heading)
                        new_cte = ((left_x + right_x) * 0.5 - img_center) / img_center
                        self._intersection_cte = (
                            (1.0 - blend) * self._intersection_cte + blend * new_cte)
                    else:
                        self._intersection_stable_count = 0

                    if self._intersection_stable_count >= self.intersect_stable_frames:
                        self._exit_intersection("recovery")
                        if lane_width > self.learned_lane_width * self.junction_width_ratio:
                            if self._straight_junction_side is None:
                                cfl = left_x + 0.50 * self.learned_lane_width
                                cfr = right_x - 0.50 * self.learned_lane_width
                                self._straight_junction_side = (
                                    'L' if abs(cfl - img_center) < abs(cfr - img_center) else 'R')
                            if self._straight_junction_side == 'L':
                                lane_center = left_x + 0.50 * self.learned_lane_width
                            else:
                                lane_center = right_x - 0.50 * self.learned_lane_width
                        else:
                            lane_center = (left_x + right_x) / 2.0
                            self._straight_junction_side = None

                        if lane_width > 0:
                            if self._width_ema_samples == 0:
                                self._width_ema = lane_width
                            else:
                                self._width_ema = 0.15 * lane_width + 0.85 * self._width_ema
                            self._width_ema_samples += 1
                        self._last_two_vec_time = now
                        self._last_good_heading = current_heading
                        raw_cte = (lane_center - img_center) / img_center
                        self._last_good_cte = raw_cte
                        self.vectors_available = True
                        self.last_vector_time = now
                        if self.learn_lane_width:
                            if 150.0 < lane_width < (img_w * 0.65):
                                self.learned_lane_width = (
                                    0.05 * lane_width + 0.95 * self.learned_lane_width)
                        self.error = max(-1.0, min(1.0, raw_cte))
                        self.target_turn = self._compute_pid(self.error, now)
                        self.target_speed = self._compute_speed(self.target_turn)
                        self.last_good_turn = self.target_turn
                        return

                    self.vectors_available = True
                    self.last_vector_time = now
                    cte = self._intersection_cte
                    turn = (self.intersect_heading_gain * self._intersection_heading
                            + self.intersect_cte_gain * cte)
                    self.target_turn = max(TURN_MIN, min(TURN_MAX, turn))
                    self.target_speed = self.intersect_speed
                    return

                else:
                    self._last_two_vec_time = now
                    if lane_width > 0:
                        if self._width_ema_samples == 0:
                            self._width_ema = lane_width
                        else:
                            self._width_ema = 0.15 * lane_width + 0.85 * self._width_ema
                        self._width_ema_samples += 1

                    if lane_width > self.learned_lane_width * self.junction_width_ratio:
                        if self._straight_junction_side is None:
                            center_from_left = left_x + 0.50 * self.learned_lane_width
                            center_from_right = right_x - 0.50 * self.learned_lane_width
                            err_l = abs(center_from_left - img_center)
                            err_r = abs(center_from_right - img_center)
                            self._straight_junction_side = 'L' if err_l < err_r else 'R'
                        if self._straight_junction_side == 'L':
                            lane_center = left_x + 0.50 * self.learned_lane_width
                        else:
                            lane_center = right_x - 0.50 * self.learned_lane_width
                    else:
                        lane_center = (left_x + right_x) / 2.0
                        self._straight_junction_side = None

                    raw_cte = (lane_center - img_center) / img_center
                    reason = None
                    if lane_width > self.learned_lane_width * self.intersect_entry_width_ratio:
                        reason = "wide"
                    elif (self._width_ema_samples
                          >= self.intersect_width_samples_for_spike
                          and self._width_ema > 0
                          and lane_width > self._width_ema * self.intersect_width_spike_ratio):
                        reason = "spike"

                    if reason is not None:
                        self._last_good_heading = current_heading
                        self._last_good_cte = raw_cte
                        self._enter_intersection(current_heading, raw_cte, reason)
                        self.vectors_available = True
                        self.last_vector_time = now
                        turn = (self.intersect_heading_gain * self._intersection_heading
                                + self.intersect_cte_gain * self._intersection_cte)
                        self.target_turn = max(TURN_MIN, min(TURN_MAX, turn))
                        self.target_speed = self.intersect_speed
                        self.last_good_turn = self.target_turn
                        return

                    self._last_good_heading = current_heading
                    self._last_good_cte = raw_cte
                    a_fast = self.intersect_fast_ema_alpha
                    a_slow = self.intersect_slow_ema_alpha
                    if not self._heading_ema_init:
                        self._heading_ema_fast = current_heading
                        self._heading_ema_slow = current_heading
                        self._heading_ema_init = True
                    else:
                        self._heading_ema_fast = a_fast * current_heading + (1.0 - a_fast) * self._heading_ema_fast
                        self._heading_ema_slow = a_slow * current_heading + (1.0 - a_slow) * self._heading_ema_slow
                    self.vectors_available = True
                    if self.learn_lane_width:
                        if 150.0 < lane_width < (img_w * 0.65):
                            self.learned_lane_width = (
                                0.05 * lane_width + 0.95 * self.learned_lane_width)
                    self.error = max(-1.0, min(1.0, raw_cte))
                    self.target_turn = self._compute_pid(self.error, now)
                    self.target_speed = self._compute_speed(self.target_turn)
                    self.last_good_turn = self.target_turn
                    return

            else:
                if self.current_mission == "LEFT":
                    offset = self._clamped_offset(0.35 * lane_width, lane_width)
                    lane_center = left_x + offset
                elif self.current_mission == "RIGHT":
                    offset = self._clamped_offset(0.35 * lane_width, lane_width)
                    lane_center = right_x - offset
                self.vectors_available = True
                if self.learn_lane_width:
                    if 150.0 < lane_width < (img_w * 0.65):
                        self.learned_lane_width = (
                            0.05 * lane_width + 0.95 * self.learned_lane_width)

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
                if (have_heading_ref
                        and not self._in_cooldown(now)
                        and heading_jump > jump_lim):
                    snap_heading = self._heading_ema_slow
                    snap_cte = self.error
                    self._enter_intersection(
                        snap_heading, snap_cte,
                        f"one_vec_jump({math.degrees(heading_jump):.0f}deg)")
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
            else:
                lane_center = aim - (lane_width - offset)
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
        self.error = max(-1.0, min(1.0, raw_error))
        self.target_turn = self._compute_pid(self.error, now)
        self.target_speed = self._compute_speed(self.target_turn)
        self.last_good_turn = self.target_turn
        if mission_straight and not self._in_intersection and count == 1:
            a_fast = self.intersect_fast_ema_alpha
            a_slow = self.intersect_slow_ema_alpha
            if not self._heading_ema_init:
                self._heading_ema_fast = one_vec_heading
                self._heading_ema_slow = one_vec_heading
                self._heading_ema_init = True
            else:
                self._heading_ema_fast = a_fast * one_vec_heading + (1.0 - a_fast) * self._heading_ema_fast
                self._heading_ema_slow = a_slow * one_vec_heading + (1.0 - a_slow) * self._heading_ema_slow

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

    # ------------------------------------------------------------------
    # LiDAR callback — obstacle avoidance + safe-zone + wall geometry
    # ------------------------------------------------------------------
    def lidar_callback(self, msg):
        if not self.obstacle_enable:
            self.obstacle_detected = False
            return

        ranges = msg.ranges
        n = len(ranges)
        if n == 0 or msg.angle_increment == 0.0:
            return

        # ---- LiDAR orientation map (debug, ~1 Hz) ----
        self._last_ranges = ranges
        self._last_range_count = n
        self._last_angle_min = msg.angle_min
        self._last_angle_increment = msg.angle_increment
        self._log_lidar_orientation()

        # ---- Obstacle avoidance sector (symmetric ±FOV/2 around 0°) ----
        half_fov = math.radians(self.obstacle_fov_deg) / 2.0
        i_center = int(round((0.0 - msg.angle_min) / msg.angle_increment))
        i_half = int(round(half_fov / abs(msg.angle_increment)))
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
        self.nearest_dist = min(left_min, right_min)

        if self.nearest_dist < self.obstacle_trigger_dist:
            self.obstacle_detected = True
        elif self.nearest_dist > self.obstacle_clear_dist:
            self.obstacle_detected = False

        if self.obstacle_detected:
            self.obstacle_turn = (self.obstacle_turn_gain
                                  if right_min > left_min else -self.obstacle_turn_gain)

        # =====================================================================
        # Safe Zone LiDAR sector (asymmetric, -45° to +15°)
        # =====================================================================
        #
        # This sector is always computed regardless of mission state so
        # that the data is fresh when the FSM enters WAITING_FOR_SAFE_ZONE.
        # Only the control_loop decides whether to *act* on it.
        #
        # Beam-density analysis:
        #   - valid beam   : finite distance > 0.05 m (not inf/NaN, not a
        #                    ground-bounce artifact)
        #   - close beam   : valid beam whose distance lies inside
        #                    [zone_close_min_dist, zone_close_max_dist]
        #   - the detector latches when the number of close beams is
        #     >= zone_close_beam_threshold for zone_confirm_scans
        #     consecutive scans.
        #
        p_min_deg = self.zone_fov_min_deg
        p_max_deg = self.zone_fov_max_deg
        i_p_min = int(round((math.radians(p_min_deg) - msg.angle_min) / msg.angle_increment))
        i_p_max = int(round((math.radians(p_max_deg) - msg.angle_min) / msg.angle_increment))
        i_p_min = max(0, min(n - 1, i_p_min))
        i_p_max = max(0, min(n - 1, i_p_max))
        valid_distances = []
        sector_data = []
        close_count = 0
        min_dist = float('inf')
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
        # Wall-parallel alignment geometry (left / right) — ENTERING_SAFE_ZONE
        # =====================================================================
        #
        # Reconstructs the mission-side wall as a local line on every scan
        # (see the module theory block).  Computed regardless of mission
        # state so the data is fresh; only ENTERING_SAFE_ZONE ACTS on it
        # (purely to feed the wall-parallel alignment steering term).
        # Normal obstacle detection is untouched.
        #
        self._update_wall_geometry(ranges, n, msg.angle_min,
                                   msg.angle_increment,
                                   getattr(msg, 'range_max', float('inf')))

    # ------------------------------------------------------------------
    # Main control loop (33 Hz)
    # ------------------------------------------------------------------
    def control_loop(self):
        now = time.time()

        # =================================================================
        # Mission assembly watchdog (atomic /target_type + /target_qr)
        # =================================================================
        accepting = (self.mission_state in (
            MissionState.NORMAL_LINE_FOLLOWING,
            MissionState.NAVIGATING_TO_NEXT_TARGET))
        if accepting and self.mission_ready:
            # Both pieces of the assignment assembled — activate
            # (defensive; the callbacks normally activate synchronously).
            self._activate_pending_mission()
        elif (accepting
              and self.pending_target_qr is not None
              and self.pending_target_type is None
              and self.pending_target_qr_time is not None
              and (now - self.pending_target_qr_time)
                  >= self.target_type_wait_timeout):
            # /target_qr arrived but /target_type never did — legacy QR
            # Detector (or a lost type message).  Activate with UNKNOWN.
            self.get_logger().warn(
                f"No /target_type within {self.target_type_wait_timeout:.1f}s "
                f"of /target_qr — activating mission with UNKNOWN "
                "(legacy) target type.")
            self._activate_pending_mission(legacy=True)

        # =================================================================
        # MISSION_COMPLETE — fully event-driven silent idle.
        #   Buggy stopped (speed=0, steering=0).  No periodic logging, no
        #   timers, no polling.  The ONLY event that wakes this state is
        #   /mission/available from the QR Detector, which immediately
        #   transitions to NAVIGATING_TO_NEXT_TARGET.
        # =================================================================
        if self.mission_state == MissionState.MISSION_COMPLETE:
            self.publish_drive_cmd(0.0, 0.0)
            return

        # =================================================================
        # WAITING_FOR_SERVER_ACK — buggy stopped, waiting for QR Detector
        # =================================================================
        if self.mission_state == MissionState.WAITING_FOR_SERVER_ACK:
            self.publish_drive_cmd(0.0, 0.0)
            if now - self._wait_log_time >= 1.0:
                self._wait_log_time = now
                self.get_logger().info(
                    f"[{self.mission_state}] "
                    "Waiting for Server Assignment...")
            return

        # =================================================================
        # PARKED_IN_SAFE_ZONE — stopped at final parking position.
        #   /safe_zone has already been published by _on_parking_complete.
        #   Silent; the ONLY wake-up is RESUME / MISSION_COMPLETE from the
        #   QR Detector (or /mission/available for the next mission).
        # =================================================================
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
                # Parking completed this tick — already stopped & published.
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
        # Driving logic (unchanged)
        # =================================================================
        if self.obstacle_detected:
            want_turn = max(TURN_MIN, min(TURN_MAX,
                            0.35 * self.target_turn + self.obstacle_turn))
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
                if self.current_mission == "STRAIGHT":
                    want_turn = self.last_good_turn * 0.5
                else:
                    want_turn = self.last_good_turn
                want_speed = self.speed_lost
            self.integral = 0.0
            self.prev_time = None

        # ---- Slow-approach / parking speed caps (target missions) ----
        # Steering and the rest of the driving logic stay untouched; only
        # the forward speed is reduced so the buggy approaches and enters
        # the safe zone accurately.
        if self.mission_state == MissionState.WAITING_FOR_SAFE_ZONE:
            want_speed = min(want_speed, self.slow_approach_speed)
        elif self.mission_state == MissionState.ENTERING_SAFE_ZONE:
            want_speed = min(want_speed, self.parking_speed)
            # PARKING DRIVE: go STRAIGHT into the parking area.  Ignore
            # line-following steering; the only lateral correction is the
            # wall-parallel alignment term applied just below.
            want_turn = 0.0

        # ---- Wall-parallel alignment steering (ENTERING_SAFE_ZONE only) ----
        # Gently rotate the buggy parallel to the parking wall while
        # driving forward, so it stops aligned with the building.  Capped
        # and inactive in every other state.
        if (self.mission_state == MissionState.ENTERING_SAFE_ZONE
                and self._wall_align_valid):
            want_turn = max(TURN_MIN, min(TURN_MAX,
                want_turn + self.wall_align_sign * self.wall_align_gain
                * self._wall_align_error))

        self.filtered_turn = (
            self.steer_alpha * want_turn + (1.0 - self.steer_alpha) * self.filtered_turn)
        self.filtered_speed = (
            self.speed_alpha * want_speed + (1.0 - self.speed_alpha) * self.filtered_speed)
        final_turn = max(TURN_MIN, min(TURN_MAX, self.filtered_turn))
        final_speed = max(SPEED_MIN, min(SPEED_MAX, self.filtered_speed))
        self.publish_drive_cmd(final_speed, self.steer_sign * final_turn)

        self._tick += 1
        if self.debug_log and self._tick % 15 == 0:
            mode = ('INT' if self._in_intersection else
                    ('OBS' if self.obstacle_detected else 'NORM'))
            self.get_logger().info(
                f"vec={'Y' if self.vectors_available else 'N'} "
                f"side={self.last_single_side} "
                f"width={self.learned_lane_width:.0f} "
                f"mission={self.current_mission} "
                f"fsm={self.mission_state} "
                f"mode={mode} "
                f"err={self.error:+.3f} "
                f"obs={'Y' if self.obstacle_detected else 'N'}@{self.nearest_dist:.2f} "
                f"turn_int={final_turn:+.3f} "
                f"joy={self.steer_sign * final_turn:+.3f} spd={final_speed:.2f}")

    # =====================================================================
    # Beam-Density Safe Zone Detector
    # =====================================================================
    def _run_safe_zone_detector(self, now):
        """Beam-density safe-zone detection.

        Called once per control-loop tick ONLY when the FSM is in
        WAITING_FOR_SAFE_ZONE.  Uses the LiDAR sector analysis from
        lidar_callback.

        Detection criterion:
          1. Every scan, count the beams in the Safe Zone sector whose
             distance is inside [zone_close_min_dist, zone_close_max_dist]
             (0.65 m – 1.00 m).  Beams that are inf/NaN or <= 0.05 m are
             not valid and never counted.
          2. If the close-beam count is >= zone_close_beam_threshold
             (6), the scan is a positive frame.
          3. The detection latches only after zone_confirm_scans (3)
             CONSECUTIVE positive scans.  A single non-positive scan
             resets the streak to 0.
          4. Duplicate publication prevention — the safe_zone_published
             flag ensures /safe_zone is published exactly ONCE per target.
        """
        # ---- Periodic debug logging (every 0.5 s) ----
        if now - self._lidar_debug_time >= 0.5:
            self._lidar_debug_time = now
            self._log_zone_debug(now)

        close_count = self._zone_close_count
        threshold = self.zone_close_beam_threshold
        if close_count >= threshold:
            # ---- Positive frame: beam-density requirement met ----
            self._zone_consecutive_scans += 1
            if self._zone_consecutive_scans >= self.zone_confirm_scans:
                self._zone_detection_state = "DETECTED"
            else:
                self._zone_detection_state = "CANDIDATE"
            self.get_logger().info(
                f"[SafeZone] {self._zone_detection_state}  "
                f"close={close_count}/{threshold} beams  "
                f"consecutive={self._zone_consecutive_scans}/{self.zone_confirm_scans}  "
                f"valid={self._zone_total_valid}  min={self._zone_min_dist:.2f}m")
            if self._zone_consecutive_scans >= self.zone_confirm_scans:
                self._on_safe_zone_detected()
        else:
            # ---- Non-positive frame: streak reset ----
            if self._zone_consecutive_scans > 0:
                self.get_logger().info(
                    f"[SafeZone] MISS — streak reset  "
                    f"close={close_count}/{threshold} beams  "
                    f"(needed {self.zone_confirm_scans} consecutive scans)")
            self._zone_consecutive_scans = 0
            self._zone_detection_state = "MISS"

    # =====================================================================
    # Safe Zone detected — enter the zone (NO /safe_zone publish yet)
    # =====================================================================
    def _on_safe_zone_detected(self):
        """Safe zone confirmed by the beam-density detector.

        Does NOT publish /safe_zone.  The buggy continues line following
        at parking speed (ENTERING_SAFE_ZONE) while the wall-parallel
        alignment detector keeps it parallel to the building and it drives
        straight for parking_forward_distance meters (odometry only).
        """
        if self._safe_zone_published:
            return
        self._zone_consecutive_scans = 0

        # Initialize the wall-parallel alignment parking bookkeeping.
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
            f"Close beams   : {self._zone_close_count} "
            f"(threshold {self.zone_close_beam_threshold})\n"
            f"Consecutive   : {self.zone_confirm_scans} scans\n"
            f"Total valid   : {self._zone_total_valid} beams\n"
            f"Min distance  : {self._zone_min_dist:.2f}m\n"
            "Entering Safe Zone...\n"
            "========================================")

        # Transition to ENTERING_SAFE_ZONE (drive straight in at parking
        # speed while maintaining wall-parallel alignment).
        self._transition_mission_state(
            MissionState.ENTERING_SAFE_ZONE,
            f"Safe zone detected (close beams "
            f"{self._zone_close_count}/{self.zone_close_beam_threshold}, "
            f"type={self.active_target_type})")

    # =====================================================================
    # LiDAR orientation map (debug, ~1 Hz)
    # =====================================================================
    def _log_lidar_orientation(self):
        """Print the beam-to-direction orientation map of the latest scan.

        Called from lidar_callback on every scan but throttled to once
        per second so the log is not flooded.  Shows beams 0, 10, 20,
        ..., 350 with their index, TRUE angle (computed from the scan's
        angle_min / angle_increment) and distance, so the FRONT / LEFT /
        RIGHT / FRONT-LEFT / FRONT-RIGHT indices can be identified in
        the simulator and the correct sector angles derived.  Purely
        informational: no control output, steering, speed, detection or
        FSM behavior is affected.
        """
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
                angle_deg = math.degrees(self._last_angle_min
                                         + idx * self._last_angle_increment)
                lines.append(
                    f"Beam {idx:3d} | Angle {angle_deg:6.1f}° | "
                    f"{fmt(ranges[idx])}")
            else:
                # Defensive: scan has fewer than 360 beams.
                lines.append(f"Beam {idx:3d} | Angle   N/A | N/A")
        lines.append("=======================================================")
        self.get_logger().info("\n".join(lines))

    # =====================================================================
    # Wall-parallel alignment geometry — computed from every scan
    # =====================================================================
    def _update_wall_geometry(self, ranges, n, angle_min, angle_increment,
                              range_max=float('inf')):
        """Reconstruct the mission-side wall as a local line on every scan.

        The fitted line feeds ONLY the wall-vs-heading alignment error used
        by the wall-parallel alignment steering term while parking.  The
        wall midpoint / length / segment-midpoint (s) values are computed
        for logging/back-compat but are NOT used in the parking stop
        decision (which is purely odometry forward distance).

        Also bumps the scan-sequence counter so the parking detector
        advances exactly once per NEW scan (the control loop runs faster
        than the LiDAR).
        """
        self._scan_seq += 1
        self._wall_geom = {}   # side -> dict or None
        for side in ("LEFT", "RIGHT"):
            self._wall_geom[side] = self._side_wall_geometry(
                ranges, n, angle_min, angle_increment, side, range_max)

    def _side_wall_geometry(self, ranges, n, angle_min, angle_increment,
                            side, range_max):
        """Wall segment reconstruction for one side half-plane.

        Returns a dict {s, len_m, lateral_m, align_err_rad, beams} or
        None when no reliable wall is visible.  All steps are
        structural/noise-filtered — none depend on the environment.
        'align_err_rad' is the only field the parking controller uses
        (for the wall-parallel alignment term); s / len_m / lateral_m
        are retained for debug logging only.
        """

        # ---- 1) contiguous solid-beam clustering (gap tolerant) ----
        # The FULL mission-side half-plane is the cluster window:
        #   RIGHT side -> bearings (0°, 180°)   (forward .. behind)
        #   LEFT  side -> bearings (-180°, 0°)
        if side == "RIGHT":
            i0 = int(round((0.0 - angle_min) / angle_increment))
            i1 = int(round((math.pi - angle_min) / angle_increment))
        else:
            i0 = int(round((-math.pi - angle_min) / angle_increment))
            i1 = int(round((0.0 - angle_min) / angle_increment))
        i0 = max(0, min(n - 1, i0))
        i1 = max(0, min(n - 1, i1))
        max_wall = self.wall_max_range_frac * range_max \
            if math.isfinite(range_max) else float('inf')

        runs = []          # list of lists of (bearing_rad, range_m)
        cur = []
        gap = 0
        for i in range(i0, i1 + 1):
            r = ranges[i % n]
            solid = (math.isfinite(r) and r > self.wall_min_range_m
                     and r < max_wall)
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

        # Main cluster = the largest contiguous face (gaps / doors split
        # the wall; the biggest piece is the parking face).
        if not runs:
            return None
        main = max(runs, key=len)
        if len(main) < self.wall_min_beams:
            return None

        # ---- 2) local Cartesian points (x forward, y mission side) ----
        pts = [(r * math.cos(a), r * math.sin(a)) for a, r in main]

        # ---- 3) robust line fit on the patch near the abeam axis ----
        side_sign = 1.0 if side == "RIGHT" else -1.0
        abeam = 0.5 * math.pi * side_sign
        patch = [p for p in pts
                 if abs(math.atan2(p[1], p[0]) - abeam)
                 <= math.radians(self.wall_fit_patch_deg)]
        if len(patch) < 3:
            patch = pts
        if len(patch) < 3:
            return None
        fit = self._fit_wall_line(patch)
        if fit is None:
            return None
        n_x, n_y, d_line = fit

        # ---- 4) project the FULL cluster -> segment, midpoint s (debug) ----
        u_x, u_y = -n_y, n_x
        if u_x < 0.0:
            u_x, u_y = -u_x, -u_y
        t_vals = []
        for (px, py) in pts:
            t_vals.append((px - d_line * n_x) * u_x
                          + (py - d_line * n_y) * u_y)
        t_min, t_max = min(t_vals), max(t_vals)

        # ---- 5) alignment error (wall direction vs heading) ----
        align_err = math.atan2(u_y, u_x)   # u_x > 0 -> in (-90, 90) deg
        return {
            "s": 0.5 * (t_min + t_max),     # (debug only)
            "len_m": t_max - t_min,         # measured wall length (debug)
            "lateral_m": d_line,            # lateral distance (debug)
            "align_err_rad": align_err,     # radians, 0 = parallel
            "beams": len(main),
        }

    def _fit_wall_line(self, pts):
        """Two-pass least-squares line fit with outlier rejection.

        Returns (n_x, n_y, d) with the unit normal n (sign such that
        n·centroid > 0) and offset d (n·p = d), or None if the fit is
        degenerate / has too few inliers.  Deterministic.
        """
        def fit_once(points):
            cx = sum(p[0] for p in points) / len(points)
            cy = sum(p[1] for p in points) / len(points)
            vxx = sum((p[0] - cx) ** 2 for p in points)
            vyy = sum((p[1] - cy) ** 2 for p in points)
            vxy = sum((p[0] - cx) * (p[1] - cy) for p in points)

            # Eigenvector of the smallest eigenvalue = line normal.
            tr = vxx + vyy
            det = vxx * vyy - vxy * vxy
            disc = math.sqrt(max(0.0, 0.25 * tr * tr - det))
            lam = 0.5 * tr - disc
            if abs(lam - vxx) > abs(lam - vyy):
                n_x, n_y = 0.0, 1.0
            else:
                n_x, n_y = 1.0, 0.0
            # (vxx - lam, vxy) is an eigenvector of the smaller eigenvalue
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
                   if abs(n_x * p[0] + n_y * p[1] - d_line)
                   <= self.wall_fit_tol_m]
        if len(inliers) < 3:
            return None
        return fit_once(inliers)

    # =====================================================================
    # Wall-parallel alignment parking detector (ENTERING_SAFE_ZONE only)
    # =====================================================================
    def _select_wall_side(self):
        """Which side defines the parking wall (for alignment only).

        LEFT/RIGHT missions: the mission side — the parking wall is by
        definition on the mission side, so the opposite building is never
        mistaken for it.  STRAIGHT missions: whichever side has a wall
        cluster.
        """
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

    def _forward_speed_estimate(self):
        """Forward speed for travelled-distance integration.

        Prefers a fresh /odom twist (linear.x) when available; otherwise
        falls back to the commanded speed (dead-reckoning).  Used only
        for the parking forward-distance measurement and the watchdog —
        never as a stop threshold derived from geometry.
        """
        if (self._odom_linear_x is not None and self._odom_time is not None
                and time.time() - self._odom_time < 0.25):
            return max(0.0, self._odom_linear_x)
        return max(0.0, self._last_cmd_speed)

    def _update_travel_distance(self, now):
        """Integrate forward speed into the travelled distance."""
        if self._last_travel_time is None:
            self._last_travel_time = now
            return
        dt = now - self._last_travel_time
        self._last_travel_time = now
        if dt <= 0.0 or dt > 0.5:
            dt = 0.033
        self._travel_dist += self._forward_speed_estimate() * dt

    def odom_callback(self, msg):
        """Optional /odom subscriber (nav_msgs/Odometry) for a better
        travelled-distance estimate.  The parking forward-distance stop is
        driven from this odometry-integrated distance."""
        try:
            self._odom_linear_x = float(msg.twist.twist.linear.x)
            self._odom_time = time.time()
        except Exception:
            pass

    def _run_parking_check(self, now):
        """Continuous wall-parallel parking — forward-distance stop.

        Called every control cycle ONLY while in ENTERING_SAFE_ZONE,
        advancing once per new scan.

        REPLACES the old wall-midpoint / s-crossing logic.  New behavior:

          * The mission-side wall is reconstructed every scan purely to
            feed the wall-parallel ALIGNMENT steering term (the buggy
            stays parallel to the building).  No wall-midpoint, wall-length
            or segment-midpoint calculation is used for the stop decision.

          * The buggy drives STRAIGHT into the parking area (the control
            loop sets want_turn = 0 in ENTERING_SAFE_ZONE, so only the
            alignment term steers).

          * The forward distance is measured with odometry only: from the
            start of ENTERING_SAFE_ZONE until travelled distance reaches
            parking_forward_distance.  When it does, the robot stops,
            publishes /safe_zone exactly once and transitions to
            PARKED_IN_SAFE_ZONE.

        A time-based failsafe watchdog still prevents endless driving.
        """
        if self._scan_seq == self._park_check_seq:
            return
        self._park_check_seq = self._scan_seq

        if self._park_start_time == 0.0:
            self._park_start_time = now
            self._park_start_dist = self._travel_dist
            self._wall_phase = "DRIVING"

        # ---- Wall side / alignment bookkeeping (alignment term only) ----
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

        # ---- Distance-based parking stop (odometry only) ----
        driven = self._travel_dist - self._park_start_dist
        if driven >= self.parking_forward_distance:
            self._on_parking_complete()
            return

        # Failsafe watchdog — TIME based, so it is not a distance tuning
        # knob and never depends on the zone geometry.
        if now - self._park_start_time > self.parking_timeout_s:
            if self.mission_state == MissionState.ENTERING_SAFE_ZONE:
                self.get_logger().warn(
                    f"Parking watchdog ({self.parking_timeout_s:.0f}s) — "
                    "parking at current position.")
            self._on_parking_complete()
            return

        self._log_parking(now)

    def _log_parking(self, now):
        """Throttled (0.5 s) wall-parallel parking debug output."""
        if now - self._parking_log_time < 0.5:
            return
        self._parking_log_time = now
        side = self._wall_side
        g = self._wall_geom.get(side) if side else None
        present = g is not None
        align_deg = math.degrees(self._wall_align_error) \
            if self._wall_align_valid else 0.0
        align_txt = f"{align_deg:+.1f} deg" if self._wall_align_valid \
            else "n/a"
        driven = self._travel_dist - self._park_start_dist
        self.get_logger().info(
            "========== PARKING (FORWARD DISTANCE) ==========\n"
            f"Side               : {side}\n"
            f"Wall present       : {'YES' if present else 'NO'}\n"
            f"Lateral distance   : {self._wall_lateral_m:.2f} m\n"
            f"Align error        : {align_txt}\n"
            f"Forward distance   : {driven:.2f} / "
            f"{self.parking_forward_distance:.2f} m\n"
            "=================================================")

    # =====================================================================
    # Parking complete — stop, publish /safe_zone, wait for QR Detector
    # =====================================================================
    def _on_parking_complete(self):
        """parking_forward_distance reached — the buggy is parked.

        THE ONLY place /safe_zone is published: robot stopped AND the
        odometry forward distance has been covered (or the watchdog
        fired).  Published exactly once per mission (guarded by
        safe_zone_published).
        """
        if self._safe_zone_published:
            return
        self._safe_zone_published = True

        # Stop the robot: linear velocity zero, no steering.
        self.publish_drive_cmd(0.0, 0.0)

        # Publish /safe_zone (exactly once per target).
        zone_msg = Bool()
        zone_msg.data = True
        self.pub_safe_zone.publish(zone_msg)

        self.get_logger().info(
            "========================================\n"
            "SAFE ZONE FULLY ENTERED\n"
            "Stopping buggy...\n"
            "Publishing /safe_zone\n"
            "Waiting for QR Detector...\n"
            "========================================")

        self._transition_mission_state(
            MissionState.PARKED_IN_SAFE_ZONE,
            "parking_forward_distance covered — /safe_zone published")

    # =====================================================================
    # LiDAR debug logging for safe-zone sector
    # =====================================================================
    def _log_zone_debug(self, now):
        """Print detailed LiDAR sector analysis for debugging.

        Shows: close beam count, total valid beams, minimum distance and
        the detection state (MISS / CANDIDATE / DETECTED).
        """
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
            f"Sector       : {self.zone_fov_min_deg:.0f}° to {self.zone_fov_max_deg:.0f}°\n"
            f"Total beams  : {total_count}\n"
            f"Valid beams  : {valid_count}\n"
            f"Close beams  : {close_count}/{threshold}  "
            f"(band {self.zone_close_min_dist:.2f}–{self.zone_close_max_dist:.2f} m)\n"
            f"Min distance : {fmt(min_dist)}\n"
            f"Detection    : {self._zone_detection_state}  "
            f"(consecutive {self._zone_consecutive_scans}/{self.zone_confirm_scans})\n"
            f"Target type  : {self.active_target_type if self.active_target_type else 'NONE'}\n"
            f"FSM state    : {self.mission_state}\n"
            "============================================")

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------
    def publish_drive_cmd(self, speed, turn):
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
        msg.axes = [0.0, float(speed), 0.0, float(turn)]
        # Remember the commanded forward speed for travelled-distance
        # dead-reckoning (parking measurement / logging).
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
