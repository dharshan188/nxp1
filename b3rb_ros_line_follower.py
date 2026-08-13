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


class MissionState:
    NORMAL_LINE_FOLLOWING = "NORMAL_LINE_FOLLOWING"
    WAITING_FOR_SAFE_ZONE = "WAITING_FOR_SAFE_ZONE"
    ENTERING_SAFE_ZONE = "ENTERING_SAFE_ZONE"
    PARKED_IN_SAFE_ZONE = "PARKED_IN_SAFE_ZONE"
    WAITING_FOR_SERVER_ACK = "WAITING_FOR_SERVER_ACK"
    NAVIGATING_TO_NEXT_TARGET = "NAVIGATING_TO_NEXT_TARGET"
    MISSION_COMPLETE = "MISSION_COMPLETE"


# This is a controller sub-mode, not a mission-FSM state. Keeping it separate
# guarantees that QR synchronization, safe-zone entry and parking transitions
# are not changed by STRAIGHT intersection guidance.
class StraightGuidanceMode:
    INACTIVE = "NORMAL_LINE_FOLLOWING"
    GUIDANCE = "STRAIGHT_GREEN_GUIDANCE"
    OBSTACLE_AVOID = "STRAIGHT_GREEN_OBSTACLE_AVOID"


class LineFollower(Node):
    def __init__(self):
        super().__init__('line_follower')

        # Parameters (all live-tunable via ros2 param set)
        self.declare_parameter('steer_sign', -1.0)
        self.declare_parameter('Kp', 0.55)
        self.declare_parameter('Ki', 0.0)
        self.declare_parameter('Kd', 0.18)
        self.declare_parameter('lookahead_blend', 0.6)
        self.declare_parameter('lookahead_frac', 0.6)
        self.declare_parameter('lane_width_px', 240.0)
        self.declare_parameter('learn_lane_width', True)
        self.declare_parameter('single_vector_side_margin', 0.20)
        self.declare_parameter('speed_straight', 0.75)
        self.declare_parameter('speed_sharp', 0.40)
        self.declare_parameter('speed_lost', 0.35)
        self.declare_parameter('steer_alpha', 0.55)
        self.declare_parameter('speed_alpha', 0.15)
        self.declare_parameter('no_vector_hold', 0.6)
        self.declare_parameter('debug_log', True)
        self.declare_parameter('obstacle_enable', True)
        self.declare_parameter('obstacle_trigger_dist', 0.90)
        self.declare_parameter('obstacle_clear_dist', 1.30)
        self.declare_parameter('obstacle_fov_deg', 60.0)
        self.declare_parameter('obstacle_turn_gain', 0.9)
        self.declare_parameter('obstacle_speed', 0.37)
        self.declare_parameter('turn_edge_margin_px', 35.0)
        self.declare_parameter('junction_width_ratio', 1.8)

        # Straight-intersection state machine
        self.declare_parameter('intersect_entry_width_ratio', 1.6)
        self.declare_parameter('intersect_exit_width_ratio', 1.25)
        self.declare_parameter('intersect_width_spike_ratio', 1.35)
        self.declare_parameter('intersect_stable_frames', 6)
        self.declare_parameter('intersect_max_time', 4.0)
        self.declare_parameter('intersect_no_vec_time', 0.18)
        self.declare_parameter('intersect_heading_gain', 0.30)
        self.declare_parameter('intersect_cte_gain', 0.20)
        self.declare_parameter('intersect_heading_blend', 0.15)
        self.declare_parameter('intersect_speed', 0.63)
        self.declare_parameter('intersect_width_samples_for_spike', 6)
        self.declare_parameter('one_vec_ff_gain', 0.75)
        self.declare_parameter('one_vec_ff_deadband_deg', 6.0)
        self.declare_parameter('one_vec_ff_max', 0.55)
        self.declare_parameter('apex_pull_gain', 0.0)
        self.declare_parameter('apex_pull_deadband_deg', 10.0)
        self.declare_parameter('apex_pull_full_deg', 35.0)
        self.declare_parameter('apex_pull_max_frac', 0.0)
        self.declare_parameter('intersect_one_vec_time', 0.25)
        self.declare_parameter('intersect_cooldown_after_timeout', 1.5)
        self.declare_parameter('intersect_max_entry_heading_deg', 30.0)
        self.declare_parameter('intersect_heading_jump_deg', 20.0)
        self.declare_parameter('intersect_memory_max_age', 1.0)
        self.declare_parameter('intersect_lock_max_heading_deg', 8.0)
        self.declare_parameter('intersect_slow_ema_alpha', 0.07)
        self.declare_parameter('intersect_fast_ema_alpha', 0.4)

        # STRAIGHT green-board guidance. These parameters affect only the
        # existing STRAIGHT intersection sub-mode; LEFT/RIGHT and parking are
        # not routed through this controller.
        self.declare_parameter('green_board_enable', True)
        self.declare_parameter(
            'green_board_topic', '/green_board_direction')
        self.declare_parameter('green_board_center_deadband', 0.03)
        self.declare_parameter('green_board_steer_gain', 0.34)
        self.declare_parameter('green_board_max_steer', 0.45)
        self.declare_parameter('green_board_hold_time', 0.60)
        self.declare_parameter('green_board_confirm_frames', 3)
        self.declare_parameter('green_board_lost_timeout', 1.50)
        self.declare_parameter('green_obstacle_recovery_enable', True)
        self.declare_parameter(
            'green_obstacle_max_avoid_angle_deg', 30.0)
        self.declare_parameter('green_obstacle_recovery_gain', 0.55)
        self.declare_parameter('green_obstacle_recovery_timeout', 3.0)
        self.declare_parameter('green_guidance_speed', 0.63)
        self.declare_parameter('green_obstacle_speed', 0.37)

        # Strict camera/LiDAR safety additions. During green guidance the
        # effective edge margin is max(turn_edge_margin_px, 25% lane width).
        # Obstacle motion requires BOTH a camera-confirmed corridor and a
        # LiDAR-confirmed clear committed side.
        self.declare_parameter('green_lane_margin_ratio', 0.25)
        self.declare_parameter('green_lane_stop_on_unsafe', True)
        self.declare_parameter('green_obstacle_min_side_clearance', 1.00)
        self.declare_parameter('green_obstacle_require_two_vectors', True)

        # Safe-zone beam-density detector
        self.declare_parameter('zone_close_min_dist', 0.65)
        self.declare_parameter('zone_close_max_dist', 1.00)
        self.declare_parameter('zone_close_beam_threshold', 6)
        self.declare_parameter('zone_confirm_scans', 3)
        self.declare_parameter('zone_fov_min_deg', -45.0)
        self.declare_parameter('zone_fov_max_deg', 15.0)

        # Slow approach + wall-parallel parking
        self.declare_parameter('slow_approach_speed', 0.32)
        self.declare_parameter('parking_speed', 0.20)
        # Patient/legacy parking remains 1.2 m. Hospital parking has its own
        # longer minimum distance and must also have a confirmed side wall.
        self.declare_parameter('parking_forward_distance', 1.2)
        self.declare_parameter('hospital_parking_forward_distance', 1.7)
        self.declare_parameter('hospital_parking_require_wall', True)
        self.declare_parameter('hospital_parking_timeout_s', 25.0)
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

        # Mission synchronization
        self.declare_parameter('target_type_wait_timeout', 0.5)

        self._reload_params()

        # Line controller runtime
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

        # Straight-intersection runtime
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

        # STRAIGHT green-board guidance runtime. Direction messages are
        # independently confirmed here even though the detector also filters
        # them, so a single transport glitch cannot flip steering.
        self.green_board_direction = "LOST"
        self._green_board_raw_direction = "LOST"
        self._green_board_candidate = None
        self._green_board_candidate_count = 0
        self._green_last_valid_direction = None
        self._green_last_valid_time = None
        self._green_guidance_mode = StraightGuidanceMode.INACTIVE
        self._green_guidance_entry_time = None
        self._green_guidance_target = "LOST"
        self._green_guidance_weight = 0.0
        self._green_last_base_turn = 0.0
        self._green_last_lane_safe = False
        self._green_debug_log_time = 0.0
        self._green_no_safe_path = False

        # Obstacle commitment/recovery state. Publicly named fields mirror the
        # requested debugging concepts; the angle is signed radians
        # (negative=LEFT, positive=RIGHT in internal image-error convention).
        self.avoidance_direction = "NONE"
        self.avoidance_start_time = None
        self.accumulated_avoidance_angle = 0.0
        self._green_avoid_target_direction = "LOST"
        self._green_obstacle_clear_since = None
        self._green_avoid_last_update_time = None
        self._green_recovery_active = False
        self._green_recovery_start_time = None
        self._green_recovery_last_update_time = None
        self._green_recovery_direction_sign = 0.0
        self._green_recovery_remaining_angle = 0.0
        self._green_recovery_initial_angle = 0.0

        # Latest EdgeVectors safety corridor. It is used only as the final
        # supervisor for STRAIGHT green guidance/avoidance; the original aim-
        # point clamping remains active for every mission.
        self._lane_safety_valid = False
        self._lane_safety_left_x = 0.0
        self._lane_safety_right_x = 0.0
        self._lane_safety_image_center = 0.0
        self._lane_safety_time = None
        self._lane_safety_source = "NONE"

        # Retain the two LiDAR side minima so a committed avoidance side can
        # be selected once and held until the obstacle-clear hysteresis wins.
        self._obstacle_left_min = float('inf')
        self._obstacle_right_min = float('inf')
        self._obstacle_scan_time = None

        # Mission FSM
        self.mission_state = MissionState.NORMAL_LINE_FOLLOWING
        self.target_qr_string = ""
        self.last_valid_mission = "NONE"

        # Safe-zone detection bookkeeping
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
        self._hospital_wall_confirmed = False
        self._parking_completion_reason = ""
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
        self._odom_angular_z = None
        self._odom_time = None

        # LiDAR debug data
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

        self._zone_valid_distances = []
        self._zone_sector_data = []
        self._wait_log_time = 0.0
        self._lidar_debug_time = 0.0

        # ROS plumbing
        self.create_subscription(
            EdgeVectors, '/edge_vectors', self.edge_vectors_callback,
            QOS_PROFILE_DEFAULT)
        self.create_subscription(
            LaserScan, '/scan', self.lidar_callback, QOS_PROFILE_DEFAULT)
        self.create_subscription(
            String, '/mission/turn', self.mission_callback,
            QOS_PROFILE_DEFAULT)
        self.subscription_green_board = self.create_subscription(
            String, self.green_board_topic, self.green_board_callback,
            QOS_PROFILE_DEFAULT)
        self.create_subscription(
            String, '/target_qr', self.target_qr_callback,
            QOS_PROFILE_DEFAULT)
        self.create_subscription(
            String, '/target_type', self.target_type_callback,
            QOS_PROFILE_DEFAULT)
        self.create_subscription(
            String, '/resume_line_following', self.resume_callback, 10)
        self.create_subscription(
            String, '/mission/available', self.mission_available_callback, 10)
        self.create_subscription(
            Odometry, '/odom', self.odom_callback, QOS_PROFILE_DEFAULT)

        self.pub_joy = self.create_publisher(
            Joy, '/cerebri/in/joy', QOS_PROFILE_DEFAULT)
        self.pub_safe_zone = self.create_publisher(
            Bool, '/safe_zone', QOS_PROFILE_DEFAULT)

        self.create_timer(0.033, self.control_loop)
        self.create_timer(1.0, self._reload_params)

        self.get_logger().info(
            "Lane-following controller loaded.\n"
            f"Mission FSM initial state: {self.mission_state}\n"
            f"STRAIGHT green-board topic: {self.green_board_topic}")

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
                f"MISSION STATE: {old} → {new_state} → {self.mission_state}"
                f"  ({reason})" if reason else "")
        else:
            self.get_logger().info(
                f"MISSION STATE: {old} → {new_state}"
                f"  ({reason})" if reason else "")

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
        self._hospital_wall_confirmed = False
        self._parking_completion_reason = ""
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

    def _reload_params(self):
        g = lambda n: self.get_parameter(n).value
        self.steer_sign = float(g('steer_sign'))
        self.Kp = g('Kp')
        self.Ki = g('Ki')
        self.Kd = g('Kd')
        self.lookahead_blend = g('lookahead_blend')
        self.lookahead_frac = g('lookahead_blend')
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
        self.intersect_width_samples_for_spike = int(
            g('intersect_width_samples_for_spike'))
        self.one_vec_ff_gain = g('one_vec_ff_gain')
        self.one_vec_ff_deadband_deg = g('one_vec_ff_deadband_deg')
        self.one_vec_ff_max = g('one_vec_ff_max')
        self.apex_pull_gain = g('apex_pull_gain')
        self.apex_pull_deadband_deg = g('apex_pull_deadband_deg')
        self.apex_pull_full_deg = g('apex_pull_full_deg')
        self.apex_pull_max_frac = g('apex_pull_max_frac')
        self.intersect_one_vec_time = g('intersect_one_vec_time')
        self.intersect_cooldown_after_timeout = g(
            'intersect_cooldown_after_timeout')
        self.intersect_max_entry_heading_deg = g(
            'intersect_max_entry_heading_deg')
        self.intersect_heading_jump_deg = g('intersect_heading_jump_deg')
        self.intersect_memory_max_age = g('intersect_memory_max_age')
        self.intersect_lock_max_heading_deg = g(
            'intersect_lock_max_heading_deg')
        self.intersect_slow_ema_alpha = g('intersect_slow_ema_alpha')
        self.intersect_fast_ema_alpha = g('intersect_fast_ema_alpha')

        self.zone_close_min_dist = g('zone_close_min_dist')
        self.zone_close_max_dist = g('zone_close_max_dist')
        self.zone_close_beam_threshold = int(g('zone_close_beam_threshold'))
        self.zone_confirm_scans = int(g('zone_confirm_scans'))
        self.zone_fov_min_deg = g('zone_fov_min_deg')
        self.zone_fov_max_deg = g('zone_fov_max_deg')

        self.slow_approach_speed = g('slow_approach_speed')
        self.parking_speed = g('parking_speed')
        self.parking_forward_distance = g('parking_forward_distance')
        self.hospital_parking_forward_distance = max(
            self.parking_forward_distance,
            float(g('hospital_parking_forward_distance')))
        self.hospital_parking_require_wall = bool(
            g('hospital_parking_require_wall'))
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
        self.hospital_parking_timeout_s = max(
            self.parking_timeout_s,
            float(g('hospital_parking_timeout_s')))
        self.target_type_wait_timeout = g('target_type_wait_timeout')

        # STRAIGHT green-board guidance parameters.
        self.green_board_enable = bool(g('green_board_enable'))
        new_green_topic = str(g('green_board_topic')).strip()
        if not new_green_topic:
            new_green_topic = '/green_board_direction'
        old_green_topic = getattr(self, 'green_board_topic', None)
        self.green_board_topic = new_green_topic
        self.green_board_center_deadband = max(
            0.0, min(0.95, float(g('green_board_center_deadband'))))
        self.green_board_steer_gain = max(
            0.0, float(g('green_board_steer_gain')))
        self.green_board_max_steer = max(
            0.0, min(1.0, float(g('green_board_max_steer'))))
        self.green_board_hold_time = max(
            0.0, float(g('green_board_hold_time')))
        self.green_board_confirm_frames = max(
            1, int(g('green_board_confirm_frames')))
        self.green_board_lost_timeout = max(
            self.green_board_hold_time,
            float(g('green_board_lost_timeout')))
        self.green_obstacle_recovery_enable = bool(
            g('green_obstacle_recovery_enable'))
        self.green_obstacle_max_avoid_angle_deg = max(
            1.0, min(60.0, float(
                g('green_obstacle_max_avoid_angle_deg'))))
        self.green_obstacle_recovery_gain = max(
            0.0, min(1.0, float(g('green_obstacle_recovery_gain'))))
        self.green_obstacle_recovery_timeout = max(
            0.1, float(g('green_obstacle_recovery_timeout')))
        self.green_guidance_speed = max(
            SPEED_MIN, min(SPEED_MAX, float(g('green_guidance_speed'))))
        self.green_obstacle_speed = max(
            SPEED_MIN, min(SPEED_MAX, float(g('green_obstacle_speed'))))
        self.green_lane_margin_ratio = max(
            0.0, min(0.45, float(g('green_lane_margin_ratio'))))
        self.green_lane_stop_on_unsafe = bool(
            g('green_lane_stop_on_unsafe'))
        self.green_obstacle_min_side_clearance = max(
            0.05, float(g('green_obstacle_min_side_clearance')))
        self.green_obstacle_require_two_vectors = bool(
            g('green_obstacle_require_two_vectors'))

        # The topic parameter is live-reloadable too. Recreate only this
        # subscription; no existing topic or callback is touched.
        if (old_green_topic is not None and
                old_green_topic != self.green_board_topic and
                hasattr(self, 'subscription_green_board')):
            self.destroy_subscription(self.subscription_green_board)
            self.subscription_green_board = self.create_subscription(
                String, self.green_board_topic,
                self.green_board_callback, QOS_PROFILE_DEFAULT)
            self.get_logger().info(
                f"Green-board subscription moved to "
                f"{self.green_board_topic}")

    # ------------------------------------------------------------------
    # Green-board direction input (String: LEFT/CENTER/RIGHT/LOST)
    # ------------------------------------------------------------------
    def green_board_callback(self, msg):
        value = msg.data.strip().upper() if msg and msg.data else "LOST"
        if value not in ("LEFT", "CENTER", "RIGHT", "LOST"):
            value = "LOST"
        self._green_board_raw_direction = value

        if value == "LOST":
            # Never manufacture a turn on a missing board. The last accepted
            # value remains timestamped for the bounded hold/decay policy.
            self.green_board_direction = "LOST"
            self._green_board_candidate = None
            self._green_board_candidate_count = 0
            return

        if value == self._green_board_candidate:
            self._green_board_candidate_count += 1
        else:
            self._green_board_candidate = value
            self._green_board_candidate_count = 1

        if self._green_board_candidate_count >= self.green_board_confirm_frames:
            self.green_board_direction = value
            self._green_last_valid_direction = value
            self._green_last_valid_time = time.time()

    # Mission synchronization callbacks
    def target_type_callback(self, msg):
        if not msg.data or not msg.data.strip():
            return
        value = msg.data.strip().upper()
        if value not in ("PATIENT", "HOSPITAL"):
            self.get_logger().info(
                f"Ignoring unknown /target_type value: {msg.data}")
            return
        if self._mission_in_progress():
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
        if self.pending_target_qr is not None and self.pending_target_type is None:
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
            self.pending_target_type = value
            self.get_logger().warn(
                f"Orphaned pending target type replaced with {value}.")

    def target_qr_callback(self, msg):
        if not msg.data or not msg.data.strip():
            return
        qr = msg.data.strip()
        if self._mission_in_progress():
            if qr == self.active_target_qr:
                self.get_logger().info(
                    f"Duplicate /target_qr ignored (active mission: {qr}).")
                if (self.pending_next_qr is None and
                        self.pending_next_type == self.active_target_type):
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
        if self.pending_target_type is not None and self.pending_target_qr is None:
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
            self.pending_target_qr = qr
            self.pending_target_qr_time = time.time()
            self.get_logger().info(
                f"Target QR received before /target_type: {qr} — "
                f"awaiting /target_type (legacy fallback after "
                f"{self.target_type_wait_timeout:.1f}s).")
        else:
            self.pending_target_qr = qr
            self.pending_target_qr_time = time.time()
            self.get_logger().warn(
                f"Pending QR replaced with {qr} (awaiting its /target_type).")

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
            self.get_logger().info(
                f"Duplicate mission ignored (already processed): "
                f"type={ttype}, QR={qr}")
            self._clear_pending_assembly()
            return
        self._last_mission_key = key
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
        if self.pending_next_type is None and self.pending_next_qr is None:
            return
        self.pending_target_type = self.pending_next_type
        self.pending_target_qr = self.pending_next_qr
        self.pending_target_qr_time = self.pending_next_qr_time
        self.pending_next_type = None
        self.pending_next_qr = None
        self.pending_next_qr_time = None
        self.get_logger().info("Pending mission promoted.")
        if (self.pending_target_type is not None and
                self.pending_target_qr is not None):
            self.mission_ready = True
            self._activate_pending_mission()

    def resume_callback(self, msg):
        received = msg.data.strip() if msg.data else ""
        if received not in ("RESUME", "MISSION_COMPLETE"):
            self.get_logger().info(f"Ignoring resume message: {received}")
            return
        self._process_resume(received)

    def _process_resume(self, received):
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
            self.get_logger().info(
                "====================================\n"
                "Waiting For New Goal Assignment...\n"
                "====================================")
            return
        self._complete_active_mission("RESUME received")
        self._transition_mission_state(
            MissionState.NAVIGATING_TO_NEXT_TARGET,
            "Server ACK / RESUME received")
        self._wait_log_time = 0.0

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
                f"Mission Topic ignored in MISSION_COMPLETE: {mission} "
                "(waiting for /mission/available)")
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
                "New mission direction while waiting for server")
            self.last_valid_mission = mission
            self.current_mission = mission
            self._straight_junction_side = None
            self._reset_intersection_state()
            self._heading_ema_init = False
            return
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
        if mission != self.current_mission:
            self.get_logger().info(f"Mission changed to {mission}")
            self.current_mission = mission
            self._straight_junction_side = None
            self._reset_intersection_state()
            self._heading_ema_init = False
        self.last_valid_mission = mission

    def mission_available_callback(self, msg):
        if not msg.data or not msg.data.strip():
            return
        assignment = msg.data.strip()
        upper = assignment.upper()
        if not (upper.startswith("PATIENT_") or
                upper.startswith("HOSPITAL_")):
            self.get_logger().info(
                f"/mission/available ignored: invalid target payload: "
                f"{assignment}")
            return
        if self.mission_state in (
                MissionState.MISSION_COMPLETE,
                MissionState.PARKED_IN_SAFE_ZONE,
                MissionState.WAITING_FOR_SERVER_ACK):
            self._complete_active_mission(
                "New mission assigned by QR Detector")
            self._transition_mission_state(
                MissionState.NAVIGATING_TO_NEXT_TARGET,
                f"New mission assigned by QR Detector ({assignment})")
            return
        self.get_logger().info(
            f"/mission/available ignored in state {self.mission_state}")

    # Geometry helpers
    def _aim_x(self, vector, curvature=0.0):
        p0, p1 = vector[0], vector[1]
        near, far = (p1, p0) if p1.y >= p0.y else (p0, p1)
        lookahead_frac = self.lookahead_frac
        abs_curve = abs(curvature)
        if 0.06 < abs_curve < 0.28:
            lookahead_frac *= 0.85
        return (1.0 - lookahead_frac) * near.x + lookahead_frac * far.x

    @staticmethod
    def _near_x(vector):
        p0, p1 = vector[0], vector[1]
        near = p1 if p1.y >= p0.y else p0
        return near.x

    @staticmethod
    def _mean_x(vector):
        return (vector[0].x + vector[1].x) / 2.0

    def _clamped_offset(self, offset, lane_width):
        margin = self.turn_edge_margin_px
        if lane_width <= 2.0 * margin:
            return lane_width / 2.0
        return max(margin, min(lane_width - margin, offset))

    def _clamp_to_lane(self, lane_center, left_edge, right_edge,
                       left_bottom=None, right_bottom=None):
        margin = self.turn_edge_margin_px
        if right_edge - left_edge <= 2.0 * margin:
            return 0.5 * (left_edge + right_edge)
        clamped = max(left_edge + margin,
                      min(right_edge - margin, lane_center))
        if left_bottom is not None and right_bottom is not None:
            left_dist = left_bottom
            right_dist = right_bottom
            edge_margin_px = 28
            if right_dist > left_dist + 2 * edge_margin_px:
                clamped = max(
                    left_dist + edge_margin_px,
                    min(right_dist - edge_margin_px, clamped))
        return clamped

    # ==================================================================
    # STRAIGHT green-board guidance and lane-safety supervisor
    # ==================================================================
    def _green_guidance_allowed(self):
        """True only for a navigational STRAIGHT intersection.

        ENTERING_SAFE_ZONE and every stopped/parking state are deliberately
        excluded so patient/hospital parking behavior cannot be changed.
        """
        return (
            self.green_board_enable and
            self.current_mission == "STRAIGHT" and
            self._in_intersection and
            self.mission_state in (
                MissionState.NORMAL_LINE_FOLLOWING,
                MissionState.NAVIGATING_TO_NEXT_TARGET,
                MissionState.WAITING_FOR_SAFE_ZONE,
            )
        )

    def _start_straight_green_guidance(self):
        if not self._green_guidance_allowed():
            return
        self._green_guidance_mode = StraightGuidanceMode.GUIDANCE
        self._green_guidance_entry_time = time.time()
        self.avoidance_direction = "NONE"
        self.avoidance_start_time = None
        self.accumulated_avoidance_angle = 0.0
        self._green_avoid_target_direction = "LOST"
        self._green_obstacle_clear_since = None
        self._green_avoid_last_update_time = None
        self._green_recovery_active = False
        self._green_recovery_start_time = None
        self._green_recovery_last_update_time = None
        self._green_recovery_direction_sign = 0.0
        self._green_recovery_remaining_angle = 0.0
        self._green_recovery_initial_angle = 0.0
        self.get_logger().info(
            "*** STRAIGHT_GREEN_GUIDANCE entered — green board selects "
            "direction; EdgeVectors remain the hard road boundary")

    def _reset_straight_green_guidance(self):
        self._green_guidance_mode = StraightGuidanceMode.INACTIVE
        self._green_guidance_entry_time = None
        self._green_guidance_target = "LOST"
        self._green_guidance_weight = 0.0
        self.avoidance_direction = "NONE"
        self.avoidance_start_time = None
        self.accumulated_avoidance_angle = 0.0
        self._green_avoid_target_direction = "LOST"
        self._green_obstacle_clear_since = None
        self._green_avoid_last_update_time = None
        self._green_recovery_active = False
        self._green_recovery_start_time = None
        self._green_recovery_last_update_time = None
        self._green_recovery_direction_sign = 0.0
        self._green_recovery_remaining_angle = 0.0
        self._green_recovery_initial_angle = 0.0
        self._green_no_safe_path = False

    def _effective_green_direction(self, now):
        """Return (last valid direction, weight) with hold then decay.

        A brief LOST uses the exact last accepted target. After hold_time the
        contribution fades to zero, and at lost_timeout the existing STRAIGHT
        intersection controller is used alone. No direction is invented.
        """
        if (self._green_last_valid_direction is None or
                self._green_last_valid_time is None):
            return "LOST", 0.0
        age = max(0.0, now - self._green_last_valid_time)
        if age <= self.green_board_hold_time:
            return self._green_last_valid_direction, 1.0
        if age >= self.green_board_lost_timeout:
            return "LOST", 0.0
        fade_span = max(
            self.green_board_lost_timeout - self.green_board_hold_time,
            1e-6)
        weight = 1.0 - (
            age - self.green_board_hold_time) / fade_span
        return self._green_last_valid_direction, max(0.0, min(1.0, weight))

    def _green_direction_bias(self, direction, weight=1.0):
        """Map board position to internal image-error steering.

        LEFT is negative and RIGHT positive, matching lane-center pixel error;
        the existing steer_sign is still applied only at final Joy publication.
        Since the topic is discrete, center_deadband is a command deadband.
        """
        scalar = {"LEFT": -1.0, "CENTER": 0.0, "RIGHT": 1.0}.get(
            direction, 0.0)
        bias = scalar * self.green_board_steer_gain * weight
        if abs(bias) <= self.green_board_center_deadband:
            bias = 0.0
        return max(
            -self.green_board_max_steer,
            min(self.green_board_max_steer, bias))

    def _update_lane_safety_bounds(self, image_width, left_x, right_x,
                                   now, source):
        if image_width <= 0.0 or right_x <= left_x:
            return
        self._lane_safety_image_center = 0.5 * image_width
        self._lane_safety_left_x = float(left_x)
        self._lane_safety_right_x = float(right_x)
        self._lane_safety_time = now
        self._lane_safety_source = source
        self._lane_safety_valid = True

    def _lane_safety_is_fresh(self, now):
        return (
            self._lane_safety_valid and
            self._lane_safety_time is not None and
            now - self._lane_safety_time <= max(0.25, self.no_vector_hold)
        )

    def _lane_clearances(self, now):
        if not self._lane_safety_is_fresh(now):
            return None
        center = self._lane_safety_image_center
        left_clear = center - self._lane_safety_left_x
        right_clear = self._lane_safety_right_x - center
        lane_width = self._lane_safety_right_x - self._lane_safety_left_x
        if lane_width <= 0.0:
            return None
        return left_clear, right_clear, lane_width

    def _green_required_lane_margin(self, lane_width):
        """Strict adaptive margin used only in STRAIGHT green guidance."""
        return max(
            1.0,
            float(self.turn_edge_margin_px),
            self.green_lane_margin_ratio * max(0.0, lane_width))

    def _green_camera_corridor_safe(self, now, require_two_vectors=False):
        """Camera must confirm a usable corridor before green motion."""
        clearances = self._lane_clearances(now)
        if clearances is None:
            return False
        if require_two_vectors:
            if self._lane_safety_source != "TWO_VECTORS":
                return False
            # Obstacle motion needs a current camera decision, not the longer
            # normal no-vector steering hold.
            if (self._lane_safety_time is None or
                    now - self._lane_safety_time > 0.20):
                return False
        left_clear, right_clear, lane_width = clearances
        margin = self._green_required_lane_margin(lane_width)
        return left_clear > margin and right_clear > margin

    def _green_lidar_side_clear(self, direction_sign, now=None):
        """Fresh LiDAR confirmation for the committed avoidance side."""
        if now is None:
            now = time.time()
        if (self._obstacle_scan_time is None or
                now - self._obstacle_scan_time > 0.25):
            return False
        distance = (
            self._obstacle_left_min if direction_sign < 0.0
            else self._obstacle_right_min)
        if math.isnan(distance):
            return False
        return (math.isinf(distance) or
                distance >= self.green_obstacle_min_side_clearance)

    def _green_joint_avoidance_safe(self, direction_sign, now):
        """Both camera corridor and LiDAR side clearance must agree."""
        if direction_sign == 0.0:
            return False
        if not self._green_camera_corridor_safe(
                now, self.green_obstacle_require_two_vectors):
            return False
        if not self._green_lidar_side_clear(direction_sign, now):
            return False
        return self._green_side_safety_scale(direction_sign, now) >= 0.10

    def _green_side_safety_scale(self, direction_sign, now):
        """0..1 permission to add steering toward one lane side."""
        clearances = self._lane_clearances(now)
        if clearances is None or direction_sign == 0.0:
            return 0.0
        left_clear, right_clear, lane_width = clearances
        clearance = left_clear if direction_sign < 0.0 else right_clear
        margin = self._green_required_lane_margin(lane_width)
        # At lane center the scale can reach one. It falls linearly to zero at
        # the strict adaptive margin, much earlier than the old fixed guard.
        center_room = max(1.0, 0.5 * lane_width - margin)
        return max(0.0, min(1.0, (clearance - margin) / center_room))

    def _apply_green_lane_safety(self, desired_turn, base_lane_turn, now):
        """Limit only the added green/obstacle contribution first."""
        if not self._lane_safety_is_fresh(now):
            # Without a fresh road boundary, never apply a blind board bias.
            return max(TURN_MIN, min(TURN_MAX, base_lane_turn)), False
        delta = desired_turn - base_lane_turn
        if abs(delta) > 1e-6:
            scale = self._green_side_safety_scale(
                -1.0 if delta < 0.0 else 1.0, now)
            desired_turn = base_lane_turn + delta * scale
        return self._enforce_green_hard_lane_safety(desired_turn, now)

    def _enforce_green_hard_lane_safety(self, turn, now):
        """FINAL direction clamp for green guidance after all controllers.

        Approaching a margin progressively removes steering toward that edge;
        entering the margin adds a bounded correction away from it. This runs
        after filtering too, so controller inertia cannot bypass the boundary.
        """
        clearances = self._lane_clearances(now)
        if clearances is None:
            return max(TURN_MIN, min(TURN_MAX, turn)), False
        left_clear, right_clear, lane_width = clearances
        margin = self._green_required_lane_margin(lane_width)
        center_room = max(1.0, 0.5 * lane_width - margin)

        if turn < 0.0:
            scale = max(
                0.0, min(1.0, (left_clear - margin) / center_room))
            turn *= scale
        elif turn > 0.0:
            scale = max(
                0.0, min(1.0, (right_clear - margin) / center_room))
            turn *= scale

        lane_safe = left_clear > margin and right_clear > margin
        if left_clear < margin:
            deficit = min(1.0, (margin - left_clear) / margin)
            correction = deficit * min(0.50, self.green_board_max_steer)
            turn = max(turn, correction)       # steer away: RIGHT
        if right_clear < margin:
            deficit = min(1.0, (margin - right_clear) / margin)
            correction = deficit * min(0.50, self.green_board_max_steer)
            turn = min(turn, -correction)      # steer away: LEFT
        return max(TURN_MIN, min(TURN_MAX, turn)), lane_safe

    def _choose_green_avoidance_sign(self, now):
        """Commit only when camera AND LiDAR approve the same side."""
        preferred = (
            1.0 if self._obstacle_right_min > self._obstacle_left_min
            else -1.0)
        alternate = -preferred
        if self._green_joint_avoidance_safe(preferred, now):
            return preferred
        if self._green_joint_avoidance_safe(alternate, now):
            return alternate
        return 0.0

    def _green_motion_delta(self, now, command, last_time_attr):
        """Measured yaw increment, with a command-integral fallback."""
        last = getattr(self, last_time_attr)
        setattr(self, last_time_attr, now)
        if last is None:
            return 0.0
        dt = now - last
        if dt <= 0.0 or dt > 0.25:
            return 0.0
        if (self._odom_angular_z is not None and self._odom_time is not None and
                now - self._odom_time < 0.25):
            return abs(self._odom_angular_z) * dt
        # Fallback is accumulated steering request, not a blind fixed timer:
        # full normalized steering is conservatively treated as 45 deg/s.
        return abs(command) * math.radians(45.0) * dt

    def _begin_green_obstacle_avoidance(self, now, target_direction):
        sign = self._choose_green_avoidance_sign(now)
        self._green_guidance_mode = StraightGuidanceMode.OBSTACLE_AVOID
        self.avoidance_start_time = now
        self.avoidance_direction = (
            "RIGHT" if sign > 0.0 else "LEFT" if sign < 0.0 else "NONE")
        self.accumulated_avoidance_angle = 0.0
        self._green_avoid_target_direction = target_direction
        self._green_obstacle_clear_since = None
        self._green_avoid_last_update_time = None
        self._green_recovery_active = False
        self._green_recovery_start_time = None
        self._green_recovery_remaining_angle = 0.0
        self._green_recovery_initial_angle = 0.0

    def _finish_green_obstacle_avoidance(self, now):
        self._green_guidance_mode = StraightGuidanceMode.GUIDANCE
        angle = abs(self.accumulated_avoidance_angle)
        avoid_sign = (
            1.0 if self.accumulated_avoidance_angle > 0.0
            else -1.0 if self.accumulated_avoidance_angle < 0.0 else 0.0)
        if (self.green_obstacle_recovery_enable and
                angle > math.radians(1.0) and avoid_sign != 0.0):
            self._green_recovery_active = True
            self._green_recovery_start_time = now
            self._green_recovery_last_update_time = None
            self._green_recovery_direction_sign = -avoid_sign
            self._green_recovery_remaining_angle = angle
            self._green_recovery_initial_angle = angle
        else:
            self._green_recovery_active = False
            self.avoidance_direction = "NONE"
            self._green_avoid_target_direction = "LOST"
        self._green_obstacle_clear_since = None

    def _compute_straight_green_control(self, now, base_turn, base_speed):
        """Combine board target, obstacle commitment/recovery and lane safety."""
        direction, weight = self._effective_green_direction(now)

        # During avoidance/recovery, preserve the target captured at obstacle
        # entry if the board becomes LOST. It is released after recovery.
        if (direction == "LOST" and
                (self._green_guidance_mode ==
                 StraightGuidanceMode.OBSTACLE_AVOID or
                 self._green_recovery_active) and
                self._green_avoid_target_direction in
                ("LEFT", "CENTER", "RIGHT")):
            direction = self._green_avoid_target_direction
            weight = 1.0

        self._green_guidance_target = direction
        self._green_guidance_weight = weight
        self._green_last_base_turn = base_turn
        self._green_no_safe_path = False

        # Strict rule: do not keep moving when camera vectors cannot prove that
        # both adaptive margins are available. This is intentionally
        # conservative—stopping is safer than touching a boundary.
        if (self.green_lane_stop_on_unsafe and
                not self._green_camera_corridor_safe(now, False)):
            self._green_no_safe_path = True
            safe_turn, lane_safe = self._enforce_green_hard_lane_safety(
                base_turn, now)
            self._green_last_lane_safe = lane_safe
            return safe_turn, 0.0

        bias = self._green_direction_bias(direction, weight)
        if direction == "LOST":
            green_desired = base_turn
        elif direction == "CENTER":
            # CENTER means straight while still retaining lane-centering.
            green_desired = 0.65 * base_turn
        else:
            # Board is primary directional reference; EdgeVectors remain the
            # bounded correction and, below, the hard safety supervisor.
            green_desired = 0.30 * base_turn + bias

        guidance_speed = min(
            self.green_guidance_speed,
            base_speed if base_speed > 0.0 else self.green_guidance_speed)
        recovery_turn = 0.0

        if self.obstacle_detected:
            if (self._green_guidance_mode !=
                    StraightGuidanceMode.OBSTACLE_AVOID):
                self._begin_green_obstacle_avoidance(now, direction)
            self._green_obstacle_clear_since = None
            avoid_sign = (
                1.0 if self.avoidance_direction == "RIGHT"
                else -1.0 if self.avoidance_direction == "LEFT" else 0.0)
            if avoid_sign == 0.0:
                # A previously unavailable corridor may become valid when a
                # fresh EdgeVectors frame arrives. Re-evaluate only while no
                # side has been committed; once chosen, never oscillate sides.
                avoid_sign = self._choose_green_avoidance_sign(now)
                if avoid_sign != 0.0:
                    self.avoidance_direction = (
                        "RIGHT" if avoid_sign > 0.0 else "LEFT")
                    self._green_avoid_last_update_time = None
            if (avoid_sign != 0.0 and
                    not self._green_joint_avoidance_safe(avoid_sign, now)):
                # A committed side is never changed mid-obstacle. If either
                # camera boundaries or LiDAR clearance stops agreeing, hold
                # position until the same side is jointly safe again.
                self._green_no_safe_path = True
                safe_turn, lane_safe = self._enforce_green_hard_lane_safety(
                    base_turn, now)
                self._green_last_lane_safe = lane_safe
                return safe_turn, 0.0
            if avoid_sign == 0.0:
                # No camera+LiDAR-confirmed side: stop for this tick. The
                # condition is re-evaluated as fresh scans/vectors arrive.
                self._green_no_safe_path = True
                safe_turn, lane_safe = self._apply_green_lane_safety(
                    base_turn, base_turn, now)
                self._green_last_lane_safe = lane_safe
                return safe_turn, 0.0

            max_angle = math.radians(
                self.green_obstacle_max_avoid_angle_deg)
            remaining = max(
                0.0, max_angle - abs(self.accumulated_avoidance_angle))
            taper = min(1.0, remaining / max(math.radians(8.0), 1e-6))
            max_normalized = min(
                1.0, self.green_obstacle_max_avoid_angle_deg / 45.0)
            lidar_magnitude = max(
                0.20, min(abs(self.obstacle_turn), max_normalized))
            avoid_turn = avoid_sign * lidar_magnitude * taper
            desired = 0.25 * green_desired + avoid_turn
            delta = self._green_motion_delta(
                now, avoid_turn, '_green_avoid_last_update_time')
            accumulated = min(
                max_angle,
                abs(self.accumulated_avoidance_angle) + delta)
            self.accumulated_avoidance_angle = avoid_sign * accumulated
            safe_turn, lane_safe = self._apply_green_lane_safety(
                desired, base_turn, now)
            self._green_last_lane_safe = lane_safe
            speed = min(self.green_obstacle_speed, self.obstacle_speed)
            return safe_turn, speed

        if (self._green_guidance_mode ==
                StraightGuidanceMode.OBSTACLE_AVOID):
            # LiDAR trigger/clear hysteresis already applies. Requiring a short
            # clear streak prevents LEFT/RIGHT re-selection on scan flicker.
            if self._green_obstacle_clear_since is None:
                self._green_obstacle_clear_since = now
            if now - self._green_obstacle_clear_since < 0.20:
                avoid_sign = (
                    1.0 if self.avoidance_direction == "RIGHT"
                    else -1.0 if self.avoidance_direction == "LEFT" else 0.0)
                if not self._green_joint_avoidance_safe(avoid_sign, now):
                    self._green_no_safe_path = True
                    safe_turn, lane_safe = (
                        self._enforce_green_hard_lane_safety(base_turn, now))
                    self._green_last_lane_safe = lane_safe
                    return safe_turn, 0.0
                settle_turn = avoid_sign * min(
                    0.25, self.green_obstacle_max_avoid_angle_deg / 90.0)
                desired = 0.50 * green_desired + settle_turn
                delta = self._green_motion_delta(
                    now, settle_turn, '_green_avoid_last_update_time')
                max_angle = math.radians(
                    self.green_obstacle_max_avoid_angle_deg)
                accumulated = min(
                    max_angle,
                    abs(self.accumulated_avoidance_angle) + delta)
                self.accumulated_avoidance_angle = avoid_sign * accumulated
                safe_turn, lane_safe = self._apply_green_lane_safety(
                    desired, base_turn, now)
                self._green_last_lane_safe = lane_safe
                return safe_turn, self.green_obstacle_speed
            self._finish_green_obstacle_avoidance(now)

        if self._green_recovery_active:
            elapsed = now - self._green_recovery_start_time
            if (self._green_recovery_remaining_angle <= math.radians(1.0) or
                    elapsed >= self.green_obstacle_recovery_timeout):
                self._green_recovery_active = False
                self._green_recovery_remaining_angle = 0.0
                self.avoidance_direction = "NONE"
                self._green_avoid_target_direction = "LOST"
            else:
                ratio = min(
                    1.0,
                    self._green_recovery_remaining_angle /
                    max(self._green_recovery_initial_angle, 1e-6))
                max_normalized = min(
                    1.0, self.green_obstacle_max_avoid_angle_deg / 45.0)
                recovery_turn = (
                    self._green_recovery_direction_sign *
                    min(max_normalized,
                        self.green_obstacle_recovery_gain * ratio))
                desired = 0.45 * green_desired + recovery_turn
                recovered = self._green_motion_delta(
                    now, recovery_turn,
                    '_green_recovery_last_update_time')
                self._green_recovery_remaining_angle = max(
                    0.0,
                    self._green_recovery_remaining_angle - recovered)
                safe_turn, lane_safe = self._apply_green_lane_safety(
                    desired, base_turn, now)
                self._green_last_lane_safe = lane_safe
                progress = 1.0 - ratio
                speed = (
                    self.green_obstacle_speed + progress *
                    (guidance_speed - self.green_obstacle_speed))
                return safe_turn, speed

        self._green_guidance_mode = StraightGuidanceMode.GUIDANCE
        safe_turn, lane_safe = self._apply_green_lane_safety(
            green_desired, base_turn, now)
        self._green_last_lane_safe = lane_safe
        return safe_turn, guidance_speed

    def _log_straight_green_guidance(self, now, steering, speed):
        if not self.debug_log or now - self._green_debug_log_time < 0.5:
            return
        self._green_debug_log_time = now
        recovery_deg = math.degrees(
            self._green_recovery_remaining_angle)
        avoid_deg = math.degrees(abs(self.accumulated_avoidance_angle))
        lane_vectors = self._lane_safety_is_fresh(now)
        avoid_sign = (
            1.0 if self.avoidance_direction == "RIGHT"
            else -1.0 if self.avoidance_direction == "LEFT" else 0.0)
        camera_safe = self._green_camera_corridor_safe(
            now,
            self.green_obstacle_require_two_vectors and
            self._green_guidance_mode ==
            StraightGuidanceMode.OBSTACLE_AVOID)
        lidar_side_safe = (
            self._green_lidar_side_clear(avoid_sign, now)
            if avoid_sign != 0.0 else not self.obstacle_detected)
        self.get_logger().info(
            "STRAIGHT GREEN GUIDANCE\n"
            "-----------------------\n"
            f"Green board : {self._green_guidance_target}\n"
            f"Lane vectors: {'YES' if lane_vectors else 'NO'} "
            f"({self._lane_safety_source})\n"
            f"Camera safe : {'YES' if camera_safe else 'NO'}\n"
            f"LiDAR safe  : {'YES' if lidar_side_safe else 'NO'}\n"
            f"Obstacle    : {'YES' if self.obstacle_detected else 'NO'}\n"
            f"Avoid dir   : {self.avoidance_direction}\n"
            f"Avoid angle : {avoid_deg:.1f}°\n"
            f"Recovery    : {recovery_deg:.1f}°\n"
            f"Steering    : {steering:+.3f}\n"
            f"Speed       : {speed:.2f}\n"
            f"Lane safe   : {'YES' if self._green_last_lane_safe else 'NO'}\n"
            f"State       : {self._green_guidance_mode}")

    # Intersection helpers
    def _reset_intersection_state(self):
        self._in_intersection = False
        self._intersection_entry_time = None
        self._intersection_heading = 0.0
        self._intersection_cte = 0.0
        self._intersection_stable_count = 0
        if hasattr(self, '_green_guidance_mode'):
            self._reset_straight_green_guidance()

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
        return 0.5 * (LineFollower._vector_heading(vec_left) +
                      LineFollower._vector_heading(vec_right))

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
                f"Intersection heading {math.degrees(heading):+.1f}deg "
                f"implausible; clamping to 0 (straight). reason={reason}")
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
        # Green steering begins only now—not on normal STRAIGHT road.
        self._start_straight_green_guidance()

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
            self._intersect_cooldown_until = (
                time.time() + self.intersect_cooldown_after_timeout)
        self._reset_intersection_state()

    # Edge vectors callback
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
            if now - self._intersection_entry_time > self.intersect_max_time:
                self._exit_intersection("timeout")

        if count >= 2:
            v1, v2 = message.vector_1, message.vector_2
            lane_heading = self._lane_heading(v1, v2)
            xa = self._aim_x(v1, lane_heading)
            xb = self._aim_x(v2, lane_heading)
            near_a = self._near_x(v1)
            near_b = self._near_x(v2)
            if self._mean_x(v1) < self._mean_x(v2):
                left_x, right_x = xa, xb
                left_x_bottom, right_x_bottom = near_a, near_b
                vec_left, vec_right = v1, v2
            else:
                left_x, right_x = xb, xa
                left_x_bottom, right_x_bottom = near_b, near_a
                vec_left, vec_right = v2, v1
            lane_width = right_x - left_x
            current_heading = self._lane_heading(vec_left, vec_right)

            # Conservative two-edge road corridor for the final green-guidance
            # supervisor. Use the inward-most lookahead/near-field boundaries;
            # fall back to lookahead edges if perspective makes it degenerate.
            safety_left = max(left_x, left_x_bottom)
            safety_right = min(right_x, right_x_bottom)
            if safety_right <= safety_left:
                safety_left, safety_right = left_x, right_x
            self._update_lane_safety_bounds(
                img_w, safety_left, safety_right, now, "TWO_VECTORS")

            if mission_straight:
                if self._in_intersection:
                    width_ok = (
                        lane_width <= self.learned_lane_width *
                        self.intersect_exit_width_ratio and lane_width > 0)
                    if width_ok:
                        self._intersection_stable_count += 1
                        blend = self.intersect_heading_blend
                        self._intersection_heading = (
                            (1.0 - blend) * self._intersection_heading +
                            blend * current_heading)
                        new_cte = (
                            (left_x + right_x) * 0.5 - img_center) / img_center
                        self._intersection_cte = (
                            (1.0 - blend) * self._intersection_cte +
                            blend * new_cte)
                    else:
                        self._intersection_stable_count = 0

                    if (self._intersection_stable_count >=
                            self.intersect_stable_frames):
                        self._exit_intersection("recovery")
                        if (lane_width > self.learned_lane_width *
                                self.junction_width_ratio):
                            if self._straight_junction_side is None:
                                cfl = left_x + 0.50 * self.learned_lane_width
                                cfr = right_x - 0.50 * self.learned_lane_width
                                self._straight_junction_side = (
                                    'L' if abs(cfl - img_center) <
                                    abs(cfr - img_center) else 'R')
                            if self._straight_junction_side == 'L':
                                lane_center = (
                                    left_x + 0.50 * self.learned_lane_width)
                            else:
                                lane_center = (
                                    right_x - 0.50 * self.learned_lane_width)
                            if abs(current_heading) > math.radians(5):
                                bias = -math.copysign(
                                    0.05 * lane_width, current_heading)
                                lane_center = self._clamped_offset(
                                    lane_center + bias, lane_width)
                        else:
                            lane_center = (left_x + right_x) / 2.0
                            self._straight_junction_side = None
                        if lane_width > 0:
                            if self._width_ema_samples == 0:
                                self._width_ema = lane_width
                            else:
                                self._width_ema = (
                                    0.15 * lane_width +
                                    0.85 * self._width_ema)
                            self._width_ema_samples += 1
                        self._last_two_vec_time = now
                        self._last_good_heading = current_heading
                        lane_center = self._clamp_to_lane(
                            lane_center, left_x, right_x,
                            left_x_bottom, right_x_bottom)
                        raw_cte = (lane_center - img_center) / img_center
                        self._last_good_cte = raw_cte
                        self.vectors_available = True
                        self.last_vector_time = now
                        if self.learn_lane_width:
                            if 150.0 < lane_width < img_w * 0.65:
                                self.learned_lane_width = (
                                    0.05 * lane_width +
                                    0.95 * self.learned_lane_width)
                        self.error = max(-1.0, min(1.0, raw_cte))
                        self.target_turn = self._compute_pid(self.error, now)
                        self.target_speed = self._compute_speed(
                            self.target_turn, current_heading)
                        self.last_good_turn = self.target_turn
                        return

                    self.vectors_available = True
                    self.last_vector_time = now
                    cte = self._intersection_cte
                    turn = (
                        self.intersect_heading_gain *
                        self._intersection_heading +
                        self.intersect_cte_gain * cte)
                    self.target_turn = max(TURN_MIN, min(TURN_MAX, turn))
                    self.target_speed = self.intersect_speed
                    return

                self._last_two_vec_time = now
                if lane_width > 0:
                    if self._width_ema_samples == 0:
                        self._width_ema = lane_width
                    else:
                        self._width_ema = (
                            0.15 * lane_width + 0.85 * self._width_ema)
                    self._width_ema_samples += 1
                if (lane_width > self.learned_lane_width *
                        self.junction_width_ratio):
                    if self._straight_junction_side is None:
                        center_from_left = (
                            left_x + 0.50 * self.learned_lane_width)
                        center_from_right = (
                            right_x - 0.50 * self.learned_lane_width)
                        err_l = abs(center_from_left - img_center)
                        err_r = abs(center_from_right - img_center)
                        self._straight_junction_side = (
                            'L' if err_l < err_r else 'R')
                    if self._straight_junction_side == 'L':
                        lane_center = (
                            left_x + 0.50 * self.learned_lane_width)
                    else:
                        lane_center = (
                            right_x - 0.50 * self.learned_lane_width)
                else:
                    lane_center = (left_x + right_x) / 2.0
                    self._straight_junction_side = None
                lane_center = self._clamp_to_lane(
                    lane_center, left_x, right_x,
                    left_x_bottom, right_x_bottom)
                raw_cte = (lane_center - img_center) / img_center
                reason = None
                if (lane_width > self.learned_lane_width *
                        self.intersect_entry_width_ratio):
                    reason = "wide"
                elif (self._width_ema_samples >=
                      self.intersect_width_samples_for_spike and
                      self._width_ema > 0 and
                      lane_width > self._width_ema *
                      self.intersect_width_spike_ratio):
                    reason = "spike"
                if reason is not None and not self._in_cooldown(now):
                    self._last_good_heading = current_heading
                    self._last_good_cte = raw_cte
                    self._enter_intersection(
                        current_heading, raw_cte, reason)
                    self.vectors_available = True
                    self.last_vector_time = now
                    turn = (
                        self.intersect_heading_gain *
                        self._intersection_heading +
                        self.intersect_cte_gain * self._intersection_cte)
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
                    self._heading_ema_fast = (
                        a_fast * current_heading +
                        (1.0 - a_fast) * self._heading_ema_fast)
                    self._heading_ema_slow = (
                        a_slow * current_heading +
                        (1.0 - a_slow) * self._heading_ema_slow)
                self.vectors_available = True
                if self.learn_lane_width:
                    if 150.0 < lane_width < img_w * 0.65:
                        self.learned_lane_width = (
                            0.05 * lane_width +
                            0.95 * self.learned_lane_width)
                self.error = max(-1.0, min(1.0, raw_cte))
                self.target_turn = self._compute_pid(self.error, now)
                self.target_speed = self._compute_speed(
                    self.target_turn, current_heading)
                self.last_good_turn = self.target_turn
                return

            junction_wide = (
                self.learned_lane_width > 0.0 and
                lane_width > self.learned_lane_width *
                self.junction_width_ratio)
            ref_width = self.learned_lane_width if junction_wide else lane_width
            if self.current_mission == "LEFT":
                offset = self._clamped_offset(0.42 * ref_width, ref_width)
                lane_center = left_x + offset
            elif self.current_mission == "RIGHT":
                offset = self._clamped_offset(0.42 * ref_width, ref_width)
                lane_center = right_x - offset
            if junction_wide:
                if self.current_mission == "LEFT":
                    lane_center = self._clamp_to_lane(
                        lane_center, left_x, left_x + ref_width)
                elif self.current_mission == "RIGHT":
                    lane_center = self._clamp_to_lane(
                        lane_center, right_x - ref_width, right_x)
            else:
                lane_center = self._clamp_to_lane(
                    lane_center, left_x, right_x)
            self.vectors_available = True
            tail_curvature = current_heading
            if self.learn_lane_width and not junction_wide:
                if 150.0 < lane_width < img_w * 0.65:
                    self.learned_lane_width = (
                        0.05 * lane_width +
                        0.95 * self.learned_lane_width)

        elif count == 1:
            v = message.vector_1
            one_vec_heading = self._vector_heading(v)
            aim = self._aim_x(v, one_vec_heading)
            mean_x = self._mean_x(v)
            band = self.side_margin * img_center
            if mean_x < img_center - band:
                self.last_single_side = 'LEFT'
            elif mean_x > img_center + band:
                self.last_single_side = 'RIGHT'
            lane_width = self.learned_lane_width
            one_vec_heading = self._vector_heading(v)

            # Reconstruct a conservative single-edge corridor using the same
            # learned lane width already used by the original controller.
            if self.last_single_side == 'LEFT':
                safety_left, safety_right = aim, aim + lane_width
            else:
                safety_left, safety_right = aim - lane_width, aim
            self._update_lane_safety_bounds(
                img_w, safety_left, safety_right, now, "ONE_VECTOR")

            if mission_straight:
                if self._in_intersection:
                    self.vectors_available = True
                    self.last_vector_time = now
                    self._intersection_stable_count = 0
                    cte = self._intersection_cte
                    turn = (
                        self.intersect_heading_gain *
                        self._intersection_heading +
                        self.intersect_cte_gain * cte)
                    self.target_turn = max(TURN_MIN, min(TURN_MAX, turn))
                    self.target_speed = self.intersect_speed
                    return
                jump_lim = math.radians(self.intersect_heading_jump_deg)
                have_heading_ref = self._heading_ema_init
                heading_jump = (
                    abs(one_vec_heading - self._heading_ema_fast)
                    if have_heading_ref else 0.0)
                if (have_heading_ref and not self._in_cooldown(now) and
                        heading_jump > jump_lim):
                    snap_heading = self._heading_ema_slow
                    snap_cte = self.error
                    self._enter_intersection(
                        snap_heading, snap_cte,
                        f"one_vec_jump({math.degrees(heading_jump):.0f}deg)")
                    self.vectors_available = True
                    self.last_vector_time = now
                    cte = self._intersection_cte
                    turn = (
                        self.intersect_heading_gain *
                        self._intersection_heading +
                        self.intersect_cte_gain * cte)
                    self.target_turn = max(TURN_MIN, min(TURN_MAX, turn))
                    self.target_speed = self.intersect_speed
                    self.last_good_turn = self.target_turn
                    return

            offset = self._clamped_offset(0.50 * lane_width, lane_width)
            apex_pull = 0.0
            outer_side = 'RIGHT' if one_vec_heading > 0.0 else 'LEFT'
            if self.last_single_side == outer_side:
                deadband = math.radians(self.apex_pull_deadband_deg)
                full = math.radians(self.apex_pull_full_deg)
                excess = abs(one_vec_heading) - deadband
                if excess > 0.0 and full > deadband:
                    strength = min(1.0, excess / (full - deadband))
                    apex_pull = min(
                        self.apex_pull_gain * strength * lane_width,
                        self.apex_pull_max_frac * lane_width)
            if self.last_single_side == 'LEFT':
                lane_center = aim + offset - apex_pull
                lane_center = self._clamp_to_lane(
                    lane_center, aim, aim + lane_width)
            else:
                lane_center = aim - (lane_width - offset) + apex_pull
                lane_center = self._clamp_to_lane(
                    lane_center, aim - lane_width, aim)
            self.vectors_available = True
            self._one_vec_ff_heading = one_vec_heading
            tail_curvature = one_vec_heading

        else:
            if mission_straight and self._in_intersection:
                self.vectors_available = True
                self.last_vector_time = now
                self._intersection_stable_count = 0
                cte = self._intersection_cte
                turn = (
                    self.intersect_heading_gain *
                    self._intersection_heading +
                    self.intersect_cte_gain * cte)
                self.target_turn = max(TURN_MIN, min(TURN_MAX, turn))
                self.target_speed = self.intersect_speed
                return
            self.vectors_available = False
            return

        self.last_vector_time = now
        raw_error = (lane_center - img_center) / img_center
        self.error = max(-1.0, min(1.0, raw_error))
        self.target_turn = self._compute_pid(self.error, now)
        ff_heading = getattr(self, '_one_vec_ff_heading', None)
        if ff_heading is not None:
            deadband = math.radians(self.one_vec_ff_deadband_deg)
            excess = abs(ff_heading) - deadband
            if excess > 0.0:
                ff = math.copysign(
                    min(self.one_vec_ff_gain * excess,
                        self.one_vec_ff_max),
                    ff_heading)
                self.target_turn = max(
                    TURN_MIN, min(TURN_MAX, self.target_turn + ff))
            self._one_vec_ff_heading = None
        self.target_speed = self._compute_speed(
            self.target_turn,
            tail_curvature if 'tail_curvature' in locals() else 0.0)
        self.last_good_turn = self.target_turn
        if mission_straight and not self._in_intersection and count == 1:
            a_fast = self.intersect_fast_ema_alpha
            a_slow = self.intersect_slow_ema_alpha
            if not self._heading_ema_init:
                self._heading_ema_fast = one_vec_heading
                self._heading_ema_slow = one_vec_heading
                self._heading_ema_init = True
            else:
                self._heading_ema_fast = (
                    a_fast * one_vec_heading +
                    (1.0 - a_fast) * self._heading_ema_fast)
                self._heading_ema_slow = (
                    a_slow * one_vec_heading +
                    (1.0 - a_slow) * self._heading_ema_slow)

    def _compute_pid(self, error, now):
        dt = 0.033 if self.prev_time is None else now - self.prev_time
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

    def _compute_speed(self, turn, curvature=0.0):
        severity = min(1.0, abs(turn))
        speed = (
            self.speed_straight - severity *
            (self.speed_straight - self.speed_sharp))
        curve_severity = min(
            1.0, abs(curvature) / math.radians(45.0))
        speed -= (
            curve_severity *
            (self.speed_straight - self.speed_sharp) * 0.6)
        return max(self.speed_sharp * 0.5, speed)

    # LiDAR callback
    def lidar_callback(self, msg):
        if not self.obstacle_enable:
            self.obstacle_detected = False
            return
        ranges = msg.ranges
        n = len(ranges)
        if n == 0 or msg.angle_increment == 0.0:
            return
        self._last_ranges = ranges
        self._last_range_count = n
        self._last_angle_min = msg.angle_min
        self._last_angle_increment = msg.angle_increment
        self._log_lidar_orientation()

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
        self._obstacle_left_min = left_min
        self._obstacle_right_min = right_min
        self._obstacle_scan_time = time.time()
        self.nearest_dist = min(left_min, right_min)
        if self.nearest_dist < self.obstacle_trigger_dist:
            self.obstacle_detected = True
        elif self.nearest_dist > self.obstacle_clear_dist:
            self.obstacle_detected = False
        if self.obstacle_detected:
            self.obstacle_turn = (
                self.obstacle_turn_gain
                if right_min > left_min else -self.obstacle_turn_gain)

        p_min_deg = self.zone_fov_min_deg
        p_max_deg = self.zone_fov_max_deg
        i_p_min = int(round(
            (math.radians(p_min_deg) - msg.angle_min) /
            msg.angle_increment))
        i_p_max = int(round(
            (math.radians(p_max_deg) - msg.angle_min) /
            msg.angle_increment))
        i_p_min = max(0, min(n - 1, i_p_min))
        i_p_max = max(0, min(n - 1, i_p_max))
        valid_distances = []
        sector_data = []
        close_count = 0
        min_dist = float('inf')
        for k in range(i_p_min, i_p_max + 1):
            idx = k % n
            r = ranges[idx]
            angle_deg = math.degrees(
                msg.angle_min + idx * msg.angle_increment)
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

        self._update_wall_geometry(
            ranges, n, msg.angle_min, msg.angle_increment,
            getattr(msg, 'range_max', float('inf')))

    # Main control loop
    def control_loop(self):
        now = time.time()
        accepting = self.mission_state in (
            MissionState.NORMAL_LINE_FOLLOWING,
            MissionState.NAVIGATING_TO_NEXT_TARGET)
        if accepting and self.mission_ready:
            self._activate_pending_mission()
        elif (accepting and self.pending_target_qr is not None and
              self.pending_target_type is None and
              self.pending_target_qr_time is not None and
              now - self.pending_target_qr_time >=
              self.target_type_wait_timeout):
            self.get_logger().warn(
                f"No /target_type within {self.target_type_wait_timeout:.1f}s "
                f"of /target_qr — activating mission with UNKNOWN "
                "(legacy) target type.")
            self._activate_pending_mission(legacy=True)

        if self.mission_state == MissionState.MISSION_COMPLETE:
            self._reset_straight_green_guidance()
            self.publish_drive_cmd(0.0, 0.0)
            return
        if self.mission_state == MissionState.WAITING_FOR_SERVER_ACK:
            self._reset_straight_green_guidance()
            self.publish_drive_cmd(0.0, 0.0)
            if now - self._wait_log_time >= 1.0:
                self._wait_log_time = now
                self.get_logger().info(
                    f"[{self.mission_state}] "
                    "Waiting for Server Assignment...")
            return
        if self.mission_state == MissionState.PARKED_IN_SAFE_ZONE:
            self._reset_straight_green_guidance()
            self.publish_drive_cmd(0.0, 0.0)
            return

        if self.mission_state == MissionState.ENTERING_SAFE_ZONE:
            # Parking remains entirely on the original controller path.
            self._reset_straight_green_guidance()
            self._update_travel_distance(now)
            self._run_parking_check(now)
            if self.mission_state == MissionState.PARKED_IN_SAFE_ZONE:
                return

        if self.mission_state == MissionState.WAITING_FOR_SAFE_ZONE:
            self._run_safe_zone_detector(now)

        if self._in_intersection:
            if (self.last_vector_time is not None and
                    now - self.last_vector_time > self.intersect_max_time):
                self._exit_intersection("watchdog")

        # Activate the green sub-mode only inside an already-detected STRAIGHT
        # intersection. All other missions and normal STRAIGHT road use the
        # original controller branches below unchanged.
        green_allowed = self._green_guidance_allowed()
        if (green_allowed and
                self._green_guidance_mode == StraightGuidanceMode.INACTIVE):
            self._start_straight_green_guidance()
        elif (not green_allowed and
              self._green_guidance_mode != StraightGuidanceMode.INACTIVE):
            self._reset_straight_green_guidance()

        green_control_active = (
            green_allowed and
            self._green_guidance_mode != StraightGuidanceMode.INACTIVE)

        if green_control_active:
            # target_turn/target_speed are the existing safe STRAIGHT
            # intersection fallback. Green guidance adds a target direction;
            # LiDAR avoidance and the final EdgeVectors supervisor are combined
            # inside this isolated controller.
            want_turn, want_speed = self._compute_straight_green_control(
                now, self.target_turn, self.target_speed)
        elif self.obstacle_detected:
            # Original obstacle behavior for LEFT/RIGHT, normal road and every
            # non-green state is preserved exactly.
            want_turn = max(
                TURN_MIN, min(
                    TURN_MAX,
                    0.35 * self.target_turn + self.obstacle_turn))
            want_speed = self.obstacle_speed
        elif self._in_intersection:
            want_turn = self.target_turn
            want_speed = self.target_speed
        elif self.vectors_available:
            want_turn = self.target_turn
            want_speed = self.target_speed
        else:
            elapsed = (
                1e9 if self.last_vector_time is None
                else now - self.last_vector_time)
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

        if self.mission_state == MissionState.WAITING_FOR_SAFE_ZONE:
            want_speed = min(want_speed, self.slow_approach_speed)
        elif self.mission_state == MissionState.ENTERING_SAFE_ZONE:
            want_speed = min(want_speed, self.parking_speed)
            if not self.vectors_available and not self.obstacle_detected:
                want_turn = self.last_good_turn

        steer_alpha = self.steer_alpha
        if abs(want_turn) > 0.35:
            steer_alpha = max(steer_alpha, 0.75)
        self.filtered_turn = (
            steer_alpha * want_turn +
            (1.0 - steer_alpha) * self.filtered_turn)
        self.filtered_speed = (
            self.speed_alpha * want_speed +
            (1.0 - self.speed_alpha) * self.filtered_speed)
        final_turn = max(TURN_MIN, min(TURN_MAX, self.filtered_turn))
        final_speed = max(SPEED_MIN, min(SPEED_MAX, self.filtered_speed))

        if green_control_active:
            # FINAL safety priority: apply after green, obstacle recovery and
            # command filtering so none of them can use filter inertia to touch
            # or cross a currently observed lane boundary.
            final_turn, lane_safe = self._enforce_green_hard_lane_safety(
                final_turn, now)
            self._green_last_lane_safe = lane_safe
            if self._green_no_safe_path:
                final_speed = 0.0
            self._log_straight_green_guidance(
                now, final_turn, final_speed)

        self.publish_drive_cmd(final_speed, self.steer_sign * final_turn)

        self._tick += 1
        if self.debug_log and self._tick % 15 == 0:
            mode = (
                'INT' if self._in_intersection else
                ('OBS' if self.obstacle_detected else 'NORM'))
            self.get_logger().info(
                f"vec={'Y' if self.vectors_available else 'N'} "
                f"side={self.last_single_side} "
                f"width={self.learned_lane_width:.0f} "
                f"mission={self.current_mission} "
                f"fsm={self.mission_state} "
                f"mode={mode} "
                f"err={self.error:+.3f} "
                f"obs={'Y' if self.obstacle_detected else 'N'}@"
                f"{self.nearest_dist:.2f} "
                f"turn_int={final_turn:+.3f} "
                f"joy={self.steer_sign * final_turn:+.3f} "
                f"spd={final_speed:.2f}")

    # Safe-zone detector
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
                f"consecutive={self._zone_consecutive_scans}/"
                f"{self.zone_confirm_scans}  "
                f"valid={self._zone_total_valid}  "
                f"min={self._zone_min_dist:.2f}m")
            if self._zone_consecutive_scans >= self.zone_confirm_scans:
                self._on_safe_zone_detected()
        else:
            if self._zone_consecutive_scans > 0:
                self.get_logger().info(
                    f"[SafeZone] MISS — streak reset  "
                    f"close={close_count}/{threshold} beams  "
                    f"(needed {self.zone_confirm_scans} consecutive scans)")
            self._zone_consecutive_scans = 0
            self._zone_detection_state = "MISS"

    def _on_safe_zone_detected(self):
        if self._safe_zone_published:
            return
        self._zone_consecutive_scans = 0
        self._wall_side = None
        self._wall_phase = "SEARCHING"
        self._wall_present_count = 0
        self._wall_miss_count = 0
        self._hospital_wall_confirmed = False
        self._parking_completion_reason = ""
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
        self._transition_mission_state(
            MissionState.ENTERING_SAFE_ZONE,
            f"Safe zone detected (close beams "
            f"{self._zone_close_count}/{self.zone_close_beam_threshold}, "
            f"type={self.active_target_type})")

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
            return "inf" if not math.isfinite(r) else f"{r:.2f} m"

        lines = ["================ LIDAR ORIENTATION MAP ================"]
        for idx in range(0, 360, 10):
            if idx < n:
                angle_deg = math.degrees(
                    self._last_angle_min +
                    idx * self._last_angle_increment)
                lines.append(
                    f"Beam {idx:3d} | Angle {angle_deg:6.1f}° | "
                    f"{fmt(ranges[idx])}")
            else:
                lines.append(f"Beam {idx:3d} | Angle   N/A | N/A")
        lines.append("=======================================================")
        self.get_logger().info("\n".join(lines))

    # Wall geometry
    def _update_wall_geometry(self, ranges, n, angle_min, angle_increment,
                              range_max=float('inf')):
        self._scan_seq += 1
        self._wall_geom = {}
        for side in ("LEFT", "RIGHT"):
            self._wall_geom[side] = self._side_wall_geometry(
                ranges, n, angle_min, angle_increment, side, range_max)

    def _side_wall_geometry(self, ranges, n, angle_min, angle_increment,
                            side, range_max):
        if side == "RIGHT":
            i0 = int(round((0.0 - angle_min) / angle_increment))
            i1 = int(round((math.pi - angle_min) / angle_increment))
        else:
            i0 = int(round((-math.pi - angle_min) / angle_increment))
            i1 = int(round((0.0 - angle_min) / angle_increment))
        i0 = max(0, min(n - 1, i0))
        i1 = max(0, min(n - 1, i1))
        max_wall = (
            self.wall_max_range_frac * range_max
            if math.isfinite(range_max) else float('inf'))
        runs = []
        cur = []
        gap = 0
        for i in range(i0, i1 + 1):
            r = ranges[i % n]
            solid = (
                math.isfinite(r) and r > self.wall_min_range_m and
                r < max_wall)
            if solid:
                cur.append((angle_min + i * angle_increment, r))
                gap = 0
            elif cur:
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
        patch = [
            p for p in pts
            if abs(math.atan2(p[1], p[0]) - abeam) <=
            math.radians(self.wall_fit_patch_deg)]
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
        t_vals = [
            (px - d_line * n_x) * u_x +
            (py - d_line * n_y) * u_y
            for px, py in pts]
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
            vxy = sum(
                (p[0] - cx) * (p[1] - cy) for p in points)
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
        inliers = [
            p for p in pts
            if abs(n_x * p[0] + n_y * p[1] - d_line) <=
            self.wall_fit_tol_m]
        if len(inliers) < 3:
            return None
        return fit_once(inliers)

    # Parking
    def _select_wall_side(self):
        g_left = self._wall_geom.get("LEFT")
        g_right = self._wall_geom.get("RIGHT")
        if self.current_mission == "LEFT":
            return "LEFT" if g_left is not None else None
        if self.current_mission == "RIGHT":
            return "RIGHT" if g_right is not None else None
        if g_left is not None and g_right is not None:
            return (
                "LEFT" if g_left["beams"] >= g_right["beams"]
                else "RIGHT")
        if g_left is not None:
            return "LEFT"
        return "RIGHT" if g_right is not None else None

    def _forward_speed_estimate(self):
        if (self._odom_linear_x is not None and
                self._odom_time is not None and
                time.time() - self._odom_time < 0.25):
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
            # Used only to measure obstacle-avoidance/recovery yaw. Parking
            # distance still uses linear.x exactly as before.
            self._odom_angular_z = float(msg.twist.twist.angular.z)
            self._odom_time = time.time()
        except Exception:
            pass

    def _is_hospital_parking(self):
        target_type = str(self.active_target_type or "").upper()
        target_qr = str(self.active_target_qr or "").upper()
        return (
            target_type == "HOSPITAL" or
            target_qr.startswith("HOSPITAL_"))

    def _active_parking_distance(self):
        return (
            self.hospital_parking_forward_distance
            if self._is_hospital_parking()
            else self.parking_forward_distance)

    def _active_parking_timeout(self):
        return (
            self.hospital_parking_timeout_s
            if self._is_hospital_parking()
            else self.parking_timeout_s)

    def _run_parking_check(self, now):
        if self._scan_seq == self._park_check_seq:
            return
        self._park_check_seq = self._scan_seq
        if self._park_start_time == 0.0:
            self._park_start_time = now
            self._park_start_dist = self._travel_dist
            self._wall_phase = "DRIVING"
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
            self._wall_present_count += 1
            if self._wall_present_count >= max(1, self.wall_confirm_frames):
                # Latch confirmation: the mission-side wall may leave the
                # LiDAR patch after the buggy has fully entered the bay.
                self._hospital_wall_confirmed = True
        else:
            self._wall_align_valid = False
            self._wall_miss_count += 1
            self._wall_present_count = 0

        hospital = self._is_hospital_parking()
        target_distance = self._active_parking_distance()
        wall_required = hospital and self.hospital_parking_require_wall
        wall_ok = (not wall_required) or self._hospital_wall_confirmed
        driven = self._travel_dist - self._park_start_dist

        # Normal hospital completion cannot happen early: BOTH 1.7 m (default)
        # and a confirmed mission-side wall are required. Patient behavior
        # remains the original distance-only rule.
        if driven >= target_distance and wall_ok:
            self._parking_completion_reason = (
                f"distance {driven:.2f}/{target_distance:.2f} m; "
                f"wall_confirmed={self._hospital_wall_confirmed}")
            self._on_parking_complete()
            return

        timeout = self._active_parking_timeout()
        if now - self._park_start_time > timeout:
            if self.mission_state == MissionState.ENTERING_SAFE_ZONE:
                self.get_logger().warn(
                    f"Parking watchdog ({timeout:.0f}s) — parking at current "
                    f"position. driven={driven:.2f}/{target_distance:.2f}m "
                    f"wall_confirmed={self._hospital_wall_confirmed}")
            self._parking_completion_reason = (
                f"watchdog {timeout:.0f}s; distance={driven:.2f}/"
                f"{target_distance:.2f}m; "
                f"wall_confirmed={self._hospital_wall_confirmed}")
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
        align_deg = (
            math.degrees(self._wall_align_error)
            if self._wall_align_valid else 0.0)
        align_txt = (
            f"{align_deg:+.1f} deg"
            if self._wall_align_valid else "n/a")
        driven = self._travel_dist - self._park_start_dist
        target_distance = self._active_parking_distance()
        hospital = self._is_hospital_parking()
        self.get_logger().info(
            "========== PARKING (FORWARD DISTANCE) ==========\n"
            f"Target type        : "
            f"{'HOSPITAL' if hospital else 'PATIENT/LEGACY'}\n"
            f"Side               : {side}\n"
            f"Wall present       : {'YES' if present else 'NO'}\n"
            f"Wall confirmed     : "
            f"{'YES' if self._hospital_wall_confirmed else 'NO'}\n"
            f"Wall streak        : {self._wall_present_count}/"
            f"{max(1, self.wall_confirm_frames)}\n"
            f"Lateral distance   : {self._wall_lateral_m:.2f} m\n"
            f"Align error        : {align_txt}\n"
            f"Forward distance   : {driven:.2f} / "
            f"{target_distance:.2f} m\n"
            "=================================================")

    def _on_parking_complete(self):
        if self._safe_zone_published:
            return
        self._safe_zone_published = True
        self.publish_drive_cmd(0.0, 0.0)
        zone_msg = Bool()
        zone_msg.data = True
        self.pub_safe_zone.publish(zone_msg)
        completion_reason = (
            self._parking_completion_reason or
            "parking distance covered")
        self.get_logger().info(
            "========================================\n"
            "SAFE ZONE FULLY ENTERED\n"
            f"Parking result: {completion_reason}\n"
            "Stopping buggy...\n"
            "Publishing /safe_zone\n"
            "Waiting for QR Detector...\n"
            "========================================")
        self._transition_mission_state(
            MissionState.PARKED_IN_SAFE_ZONE,
            f"{completion_reason} — /safe_zone published")

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
            return 'inf' if not math.isfinite(r) else f'{r:.2f}'

        self.get_logger().info(
            "================ LIDAR DEBUG ================\n"
            f"Sector       : {self.zone_fov_min_deg:.0f}° to "
            f"{self.zone_fov_max_deg:.0f}°\n"
            f"Total beams  : {total_count}\n"
            f"Valid beams  : {valid_count}\n"
            f"Close beams  : {close_count}/{threshold}  "
            f"(band {self.zone_close_min_dist:.2f}–"
            f"{self.zone_close_max_dist:.2f} m)\n"
            f"Min distance : {fmt(min_dist)}\n"
            f"Detection    : {self._zone_detection_state}  "
            f"(consecutive {self._zone_consecutive_scans}/"
            f"{self.zone_confirm_scans})\n"
            f"Target type  : "
            f"{self.active_target_type if self.active_target_type else 'NONE'}\n"
            f"FSM state    : {self.mission_state}\n"
            "============================================")

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
