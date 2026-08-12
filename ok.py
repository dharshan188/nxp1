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
#   BONUS_PARKING_TEST        — TEMPORARY DEBUG state.  Activated ONLY
#                              after the FIRST HOSPITAL has been
#                              completed and /mission/available has
#                              arrived.  In this state the robot moves
#                              forward slowly, watches /edge_vectors for
#                              one side of the lane to disappear, asks
#                              /scan to confirm the corresponding side
#                              is clear, then steers into the opening.
#                              No cone detection, no OpenCV, no precise
#                              alignment.  Isolated from the rest of the
#                              FSM.  This state was added incrementally
#                              for the first-hospital test; it is the
#                              ONLY addition to the existing FSM.
#
#                              CRITICAL FLOW REQUIREMENT: while in
#                              BONUS_PARKING_TEST the FSM is treated as
#                              BUSY (see _mission_in_progress()), so any
#                              /target_type or /target_qr that arrives
#                              for the next mission is queued in
#                              pending_next_* — never lost.  The mission
#                              that triggered the bonus entry is ALSO
#                              mirrored into pending_next_* by
#                              mission_available_callback as soon as it
#                              arrives, so even if /target_type and
#                              /target_qr never come, the assignment is
#                              preserved.  When the bonus test exits
#                              (forward-distance success), the FSM
#                              transitions to NAVIGATING_TO_NEXT_TARGET,
#                              _promote_pending_next() is invoked, and
#                              the preserved assignment is activated.
#                              On safety abort the FSM transitions to
#                              MISSION_COMPLETE — the next assignment
#                              stays in pending_next_* and will be
#                              promoted on the next /mission/available
#                              cycle.
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
# BONUS PARKING state machine — explicit state names
# =====================================================================
#
# The BONUS_PARKING_TEST state contains an explicit, named state
# machine that follows a simple, human-like parking pattern.
#
#   BONUS_PARKING_TEST
#       |
#       v
#   BONUS_SEARCH
#       Drive forward slowly while watching /edge_vectors.
#       Require a stable single-side indication for
#       `bonus_missing_side_frames` consecutive ticks.  When the
#       streak is reached, LATCH the missing side as the
#       parking side (LEFT or RIGHT) and transition to
#       BONUS_APPROACH.  The side is NEVER re-evaluated after
#       latching.
#       |
#       v
#   BONUS_APPROACH
#       Drive STRAIGHT forward for approximately
#       `bonus_park_approach_distance_m` (default ~0.30 m) at
#       `bonus_park_approach_speed` (default ~0.15 m/s) with
#       steering = 0.  This is the small "move slightly ahead"
#       step that physically positions the buggy before the
#       full turn begins.
#       |
#       v
#   BONUS_FULL_TURN
#       Drive forward slowly while turning STRONGLY toward the
#       latched side for approximately
#       `bonus_park_full_turn_max_duration_s` (default 4.0 s)
#       OR until the FRONT is critically close, whichever
#       comes first.  Steering is a fixed strong value
#       (`bonus_park_full_turn_strength`, default ~0.80) in
#       the latched-side direction.  No lane-vector or
#       proportional LiDAR steering is applied — this is a
#       deliberate, deterministic turn.
#       |
#       v
#   BONUS_STRAIGHTEN
#       Steering is set to ZERO but the buggy continues moving
#       forward for approximately
#       `bonus_park_straighten_duration_s` (default 0.4 s).
#       This is the TURN -> STEERING = 0 -> FORWARD transition
#       that allows the buggy to become parallel inside the
#       slot.  Speed is
#       `bonus_park_approach_speed` (default ~0.15 m/s).
#       |
#       v
#   BONUS_PARK_FORWARD
#       Drive STRAIGHT forward (steering = 0) for approximately
#       `bonus_park_forward_distance_m` (default 0.75 m) at
#       `bonus_park_forward_speed` (default ~0.15 m/s).  This
#       is the final "drive deeper into the parking slot"
#       motion that produces the success condition.
#       |
#       v
#   BONUS_PARKED  (terminal)
#       Robot stopped.  Linear velocity = 0, steering = 0.
#
# Recovery sub-states (entered from any active parking state
# on a genuine FRONT-collision reading — EMERGENCY/BLOCKED/
# CRITICAL — via `_bonus_simple_safety`):
#
#   BONUS_RECOVERY_STOP
#       Publish zero velocity for ~0.2 s so the LiDAR
#       scan can catch up.
#       |
#       v
#   BONUS_RECOVERY_REVERSE
#       Drive backwards at ~-0.08 m/s with steering = 0
#       for a configurable duration (default ~1.0 s).
#       |
#       v
#   BONUS_RECOVERY_STRAIGHTEN
#       Stop, re-read LiDAR, optionally apply a small
#       corrective steering nudge for ~0.3 s.
#       |
#       v
#   BONUS_RECOVERY_RETRY
#       Drive forward straight for the RETRY approach
#       distance, then transition back to BONUS_FULL_TURN
#       on the SAME latched side.  The retry physically
#       changes the starting pose from the previous
#       attempt.
#
# Notes:
#   * The latched parking side (LEFT / RIGHT) is tracked in
#     `_bonus_latched_side`.  The visible phase name in
#     `_bonus_phase` becomes "BONUS_APPROACH" /
#     "BONUS_FULL_TURN" / "BONUS_STRAIGHTEN" /
#     "BONUS_PARK_FORWARD" / "BONUS_PARKED" etc. — there
#     are NO separate "PARKING_LEFT" / "PARKING_RIGHT"
#     phase names.
#   * The recovery sub-states are stored in
#     `_bonus_recovery_step` and the externally-visible
#     `_bonus_phase` stays in the current active parking
#     phase so the operator can see which "active" phase
#     is being recovered.
#   * The watchdog (`bonus_parking_timeout_s`, default 60 s)
#     only fires when the WHOLE state machine has been running
#     for longer than the timeout AND no recovery sub-state is
#     in progress.  Recovery sub-states are allowed to take
#     as long as they need to bring the buggy back to a safe
#     position.
#   * LiDAR is used ONLY for genuine forward collision safety
#     and recovery.  Side cones are EXPECTED during parking and
#     NEVER cause a STOP or recovery cycle.
#
# =====================================================================

class BonusPhase:
    """Explicit state names for the BONUS_PARKING_TEST FSM.

    The state machine is intentionally simple (per user spec) and
    follows a human-like parking pattern:

        BONUS_SEARCH
            Drive forward while watching /edge_vectors.  Require a
            stable single-side indication for
            `bonus_missing_side_frames` consecutive ticks.  When the
            streak is reached, LATCH the missing side as the
            parking side (LEFT or RIGHT) and transition to
            BONUS_APPROACH.  The side is NEVER re-evaluated after
            latching.

        BONUS_APPROACH
            Drive STRAIGHT forward for approximately
            `bonus_park_approach_distance_m` (default ~0.30 m) at
            `bonus_park_approach_speed` (default ~0.15 m/s) with
            steering = 0.  This is the small "move slightly ahead"
            step that physically positions the buggy before the
            full turn begins.

        BONUS_FULL_TURN
            Drive forward slowly while turning STRONGLY toward the
            latched side for approximately
            `bonus_park_full_turn_max_duration_s` (default 4.0 s)
            OR until the FRONT is critically close, whichever
            comes first.  Steering is a fixed strong value
            (`bonus_park_full_turn_strength`, default ~0.80) in
            the latched-side direction.  No lane-vector or
            proportional LiDAR steering is applied — this is a
            deliberate, deterministic turn.

        BONUS_STRAIGHTEN
            Steering is set to ZERO but the buggy continues moving
            forward for approximately
            `bonus_park_straighten_duration_s` (default 0.4 s).
            This is the TURN -> STEERING = 0 -> FORWARD transition
            that allows the buggy to become parallel inside the
            slot.  Speed is
            `bonus_park_approach_speed` (default ~0.15 m/s).

        BONUS_PARK_FORWARD
            Drive STRAIGHT forward (steering = 0) for approximately
            `bonus_park_forward_distance_m` (default 0.75 m) at
            `bonus_park_forward_speed` (default ~0.15 m/s).  This
            is the final "drive deeper into the parking slot"
            motion that produces the success condition.

        BONUS_PARKED  (terminal)
            Declared when BONUS_PARK_FORWARD has covered the
            required distance.  Publish zero velocity, log
            "BONUS PARKING SUCCESS", stay stopped.

        BONUS_FAILED  (terminal)
            Declared when the attempt counter exceeds
            `bonus_park_max_attempts` or the overall watchdog
            fires.  Publish zero velocity, log
            "BONUS PARKING FAILED".

        Recovery sub-states (entered from any active parking state
        on a genuine FRONT-collision reading — EMERGENCY/BLOCKED/
        CRITICAL — via `_bonus_simple_safety`):

            BONUS_RECOVERY_STOP
                Publish zero velocity for ~0.2 s so the LiDAR
                scan can catch up.

            BONUS_RECOVERY_REVERSE
                Drive backwards at ~-0.08 m/s with steering = 0
                for a configurable duration (default ~1.0 s).

            BONUS_RECOVERY_STRAIGHTEN
                Stop, re-read LiDAR, optionally apply a small
                corrective steering nudge for ~0.3 s.

            BONUS_RECOVERY_RETRY
                Drive forward straight for the RETRY approach
                distance, then transition back to BONUS_FULL_TURN
                on the SAME latched side.  The retry physically
                changes the starting pose from the previous
                attempt.

    Every state has an explicit handler in `_run_bonus_tick` — no
    "unknown phase" fallback is ever reached.
    """
    SEARCH = "BONUS_SEARCH"
    APPROACH = "BONUS_APPROACH"
    FULL_TURN = "BONUS_FULL_TURN"
    STRAIGHTEN = "BONUS_STRAIGHTEN"
    PARK_FORWARD = "BONUS_PARK_FORWARD"
    PARKED = "BONUS_PARKED"
    FAILED = "BONUS_FAILED"

    # Recovery sub-states (used both as `BonusPhase` constants and
    # as `BonusRecovery.*` aliases for backward compatibility).
    #
    # The new recovery sequence (per spec rev 2) is:
    #   STOP -> REVERSE (with COUNTER-STEER)
    #        -> STOP2
    #        -> STRAIGHTEN  (short forward run to re-align)
    #        -> STOP3
    #        -> FORWARD     (0.65 m straight, to a NEW starting pose)
    #        -> back to FULL_TURN (with the SAME latched side).
    #
    # The split into multiple STOP / STRAIGHTEN phases lets the
    # operator see in the logs exactly where the recovery is and
    # gives the LiDAR a clean re-read between each sub-step.
    RECOVERY_STOP = "BONUS_RECOVERY_STOP"
    RECOVERY_REVERSE = "BONUS_RECOVERY_REVERSE"
    RECOVERY_STOP2 = "BONUS_RECOVERY_STOP2"           # post-reverse pause
    RECOVERY_STRAIGHTEN = "BONUS_RECOVERY_STRAIGHTEN"  # short straighten
    RECOVERY_STOP3 = "BONUS_RECOVERY_STOP3"           # post-straighten pause
    RECOVERY_FORWARD = "BONUS_RECOVERY_FORWARD"       # 0.65 m forward
    RECOVERY_RETRY = "BONUS_RECOVERY_RETRY"           # legacy alias


class BonusRecovery:
    """Aliases for the recovery sub-states.

    The recovery cycle (per spec rev 2) is:

        IDLE                  — normal parking control
        STOP -> REVERSE -> STOP2 -> STRAIGHTEN -> STOP3 -> FORWARD
            -> back to FULL_TURN (with the SAME latched side)

    The REVERSE step uses COUNTER-STEERING (LEFT parking reverses
    with a RIGHT turn, RIGHT parking reverses with a LEFT turn) so
    the buggy physically rotates toward a better starting
    orientation for the next parking attack.  This replaces the
    old "reverse straight with steering=0" behaviour which only
    moved the buggy backward without changing its heading.

    These names are kept as aliases of `BonusPhase.RECOVERY_*` so
    any external code that referenced `BonusRecovery.STOP` etc.
    still works.
    """
    IDLE = "IDLE"
    STOP = BonusPhase.RECOVERY_STOP
    REVERSE = BonusPhase.RECOVERY_REVERSE
    STOP2 = BonusPhase.RECOVERY_STOP2
    STRAIGHTEN = BonusPhase.RECOVERY_STRAIGHTEN
    STOP3 = BonusPhase.RECOVERY_STOP3
    FORWARD = BonusPhase.RECOVERY_FORWARD
    REALIGN = BonusPhase.RECOVERY_STRAIGHTEN   # legacy alias
    RETRY = BonusPhase.RECOVERY_RETRY         # legacy alias


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
    BONUS_PARKING_TEST       = "BONUS_PARKING_TEST"
    MISSION_COMPLETE         = "MISSION_COMPLETE"


def _infer_target_type_from_qr(qr: str) -> str:
    """Infer the target type ("PATIENT" / "HOSPITAL" / "UNKNOWN") from a
    QR identifier like "PATIENT_2" or "HOSPITAL_1".

    Used by mission_available_callback to synthesise a pending_next_type
    when the QR Detector publishes /mission/available BEFORE
    /target_type for the same assignment.  This guarantees the next
    mission is never lost, even if the type/QR callbacks never fire
    again.
    """
    if not qr:
        return "UNKNOWN"
    q = qr.strip().upper()
    if q.startswith("PATIENT_"):
        return "PATIENT"
    if q.startswith("HOSPITAL_"):
        return "HOSPITAL"
    return "UNKNOWN"


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
        self.declare_parameter('intersect_speed', 0.63)
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
        self.declare_parameter('slow_approach_speed', 0.32)
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
        # Maximum age of the LAST observed /target_qr read that we
        # still trust as a "current hospital" indicator for the
        # hospital-mismatch guard in _on_safe_zone_detected.  If the
        # last QR is older than this, a wrong-hospital read is
        # considered too stale to override the LiDAR close-beam
        # detection, so the buggy commits to the stop.  If the last
        # QR is fresher than this and it does NOT match the assigned
        # hospital, the buggy keeps moving.
        self.declare_parameter('hospital_qr_max_age_s', 3.0)

        # =====================================================================
        # Closed-loop LiDAR safety / parking-maneuver parameters:
        #
        #   bonus_park_front_fov_deg      FRONT sector half-width (deg)
        #   bonus_park_frontleft_lo_deg   FRONT-LEFT sector: lower (most
        #                                 negative) angle in deg
        #   bonus_park_frontleft_hi_deg   FRONT-LEFT sector: upper (less
        #                                 negative) angle in deg
        #   bonus_park_left_lo_deg        LEFT sector: lower (most
        #                                 negative) angle in deg
        #   bonus_park_left_hi_deg        LEFT sector: upper (less
        #                                 negative) angle in deg
        #
        #   The three thresholds (from large/safe to small/dangerous):
        #
        #     caution_m   — the robot is approaching an obstacle.
        #                   Slow down and reduce the turn.
        #     critical_m  — the obstacle is too close.  Stop
        #                   forward motion and straighten /
        #                   counter-steer.
        #     emergency_m — the obstacle is critically close.
        #                   Immediate STOP (zero velocity, zero turn).
        #     blocked_m   — a sector whose min is below this is
        #                   considered "blocked" and triggers a full
        #                   recovery cycle (STOP -> REVERSE -> ALIGN
        #                   -> RETRY).
        #
        #   These are intentionally MORE CONSERVATIVE than the
        #   physical cone distance, so the robot starts braking and
        #   straightening BEFORE the cone is reached.  Default values
        #   assume a buggy that can stop from full turn-speed inside
        #   ~0.8 m; tighten them if the cone is closer.
        #
        #   bonus_park_reverse_speed      small negative speed used in
        #                                 the REVERSE recovery step
        #   bonus_park_reverse_distance_m minimum reverse distance
        #                                 before the ALIGN step fires
        #   bonus_park_reverse_duration_s max reverse duration (whichever
        #                                 of distance or time fires first
        #                                 ends REVERSE)
        #   bonus_park_align_pause_s      pause between REVERSE and
        #                                 the next forward attempt
        #   bonus_park_align_min_clearance_m
        #                                 required left/front-left
        #                                 clearance before a retry is
        #                                 allowed to start the LEFT
        #                                 turn again
        #   bonus_park_max_parking_retries
        #                                 max number of full ATTEMPTS
        #                                 (each retry STARTS a fresh
        #                                 TURN with a smaller turn and
        #                                 lower speed).  When this
        #                                 counter is exceeded the
        #                                 controller enters
        #                                 BONUS_PARKING_FAILED, NOT
        #                                 BONUS_PARKED.
        #   bonus_park_turn_reduction_per_retry
        #                                 per-retry scale reduction
        #                                 applied to the turn target
        #                                 (e.g. 0.25 -> attempt 1 is
        #                                 full turn, attempt 2 is 75%
        #                                 of full, attempt 3 is 50%,
        #                                 etc., clamped to
        #                                 bonus_park_min_turn_scale)
        #   bonus_park_speed_reduction_per_retry
        #                                 per-retry scale reduction
        #                                 applied to the forward speed
        #                                 (clamped to
        #                                 bonus_park_min_speed_scale)
        #   bonus_park_min_turn_scale     floor on the turn scale
        #   bonus_park_min_speed_scale    floor on the speed scale
        #   bonus_park_min_openings       min number of finite beams
        #                                 required to declare a sector
        #                                 "open" (so a few stray inf
        #                                 returns don't cause a false
        #                                 "open" reading)
        #   bonus_park_enter_distance_m    forward distance covered
        #                                 while in BONUS_PARKING_FORWARD
        #                                 before transitioning to
        #                                 BONUS_PARKED (the SUCCESS
        #                                 condition)
        # =====================================================================
        self.declare_parameter('bonus_enable', True)
        self.declare_parameter('bonus_speed', 0.12)
        self.declare_parameter('bonus_turn_speed', 0.15)
        self.declare_parameter('bonus_turn_gain', 0.85)
        self.declare_parameter('bonus_straighten_gain', 0.25)
        self.declare_parameter('bonus_missing_side_frames', 5)
        self.declare_parameter('bonus_clear_confirm_frames', 3)
        self.declare_parameter('bonus_side_offset_deg', 30.0)
        self.declare_parameter('bonus_clear_fov_deg', 35.0)
        self.declare_parameter('bonus_side_clear_m', 1.0)
        self.declare_parameter('bonus_side_min_beams', 4)
        self.declare_parameter('bonus_side_block_m', 0.45)
        self.declare_parameter('bonus_enter_distance_m', 1.4)
        self.declare_parameter('bonus_max_duration_s', 25.0)
        self.declare_parameter('bonus_turn_slew', 0.05)
        # Per spec rev 2 the FULL_TURN must reach the FULL
        # steering command (LEFT: +1.00, RIGHT: -1.00).  The
        # old 0.9 cap was an arbitrary safety clamp; raise it
        # to 1.0 so the FULL_TURN handler can deliver the
        # commanded ±1.00 sign convention.
        self.declare_parameter('bonus_turn_max', 1.0)
        # Closed-loop maneuver parameters — conservative defaults so
        # the robot starts braking BEFORE the cone.
        self.declare_parameter('bonus_park_front_fov_deg', 20.0)
        self.declare_parameter('bonus_park_frontleft_lo_deg', -70.0)
        self.declare_parameter('bonus_park_frontleft_hi_deg', -20.0)
        self.declare_parameter('bonus_park_left_lo_deg', -120.0)
        self.declare_parameter('bonus_park_left_hi_deg', -70.0)
        # FRONT-RIGHT and RIGHT sectors are mirror-symmetric to
        # FRONT-LEFT and LEFT across the FRONT axis.  Positive
        # angles map to the robot's RIGHT side, negative to the
        # LEFT side.  These are used when the missing side is
        # RIGHT (so the robot turns RIGHT and the relevant
        # safety sectors are the right half-plane).
        self.declare_parameter('bonus_park_frontright_lo_deg', +20.0)
        self.declare_parameter('bonus_park_frontright_hi_deg', +70.0)
        self.declare_parameter('bonus_park_right_lo_deg', +70.0)
        self.declare_parameter('bonus_park_right_hi_deg', +120.0)
        # Side-directed turn strength for BONUS_PARKING_ENTRY.
        # This is the BASE turn target applied immediately when
        # the FSM enters ENTRY.  In the joystick convention
        # used by this vehicle, positive `axes[3]` = physical
        # LEFT and negative `axes[3]` = physical RIGHT (this
        # was verified experimentally — the normal line
        # follower publishes `self.steer_sign * final_turn` with
        # `steer_sign = -1.0` and `final_turn > 0` for "the
        # buggy is too far LEFT, turn right", which becomes
        # `axes[3] < 0`).  So the physical convention is:
        #   axes[3] > 0 -> physical LEFT
        #   axes[3] < 0 -> physical RIGHT
        # This is the OPPOSITE of the natural "turn_cmd > 0 =
        # turn right" convention.  The bonus parking controller
        # follows the joystick convention directly — see
        # `_bonus_command_left()` / `_bonus_command_right()`
        # helpers below.
        self.declare_parameter('bonus_park_entry_turn_strength', 0.70)
        # Forward speed used during PARKING_LEFT / PARKING_RIGHT.
        # Slow enough to allow a visible turn, fast enough to
        # actually move into the gap within a few seconds.
        self.declare_parameter('bonus_park_entry_speed', 0.05)
        # LiDAR thresholds (per spec).  Only the FRONT sector
        # determines whether the buggy must stop — the side
        # sectors are EXPECTED to report small distances (the
        # buggy is intentionally moving toward the parking
        # side) and are used only for an "avoidance nudge",
        # not for stopping.
        self.declare_parameter('bonus_park_front_emergency_m', 0.20)
        # Per spec rev 2: FRONT blocked threshold bumped DOWN
        # from 0.30 m to 0.25 m so recovery does not fire too
        # early while the buggy is still well clear.
        self.declare_parameter('bonus_park_front_blocked_m', 0.25)
        # Success clearance kept at 0.45 m — the success log
        # still requires the front to be at least this far
        # clear when the 1.30 m forward target is reached.
        self.declare_parameter('bonus_park_success_clearance_m', 0.45)
        # How long to reverse on each recovery attempt
        # (time-based, NOT distance-based, because odometry
        # may not report a negative linear.x reliably in
        # all simulators).  The reverse must actually happen,
        # so the time fallback is the primary terminator.
        self.declare_parameter('bonus_park_reverse_speed', -0.06)
        self.declare_parameter('bonus_park_reverse_duration_s', 1.5)
        self.declare_parameter('bonus_park_reverse_min_distance_m', 0.10)
        self.declare_parameter('bonus_park_align_duration_s', 0.4)
        # Stop dwell before reverse (lets the LiDAR scan catch
        # up before we start moving backward).
        self.declare_parameter('bonus_park_stop_dwell_s', 0.2)
        # NOTE: bonus_park_max_attempts is declared ONCE below
        # in the new explicit state-machine parameter block
        # (default 6, per spec rev 2).  The legacy declaration
        # with default 4 was removed to avoid
        # `rclpy.exceptions.ParameterAlreadyDeclaredException`
        # at node startup.
        # Parking-success travel distance: the robot must physically
        # travel at least this far into the missing lane after the
        # parking side is latched.  Default 0.80 m ≈ one full lane
        # width so BONUS_PARKED only fires after the robot has
        # ACTUALLY entered the parking gap, not merely detected it.
        # Override via ros2 param if the test course uses a
        # different lane width.
        self.declare_parameter('bonus_park_enter_distance_m', 0.80)

        # =====================================================================
        # Closed-loop bonus parking — new explicit state-machine parameters
        # =====================================================================
        #   bonus_parking_timeout_s              overall watchdog (replaces
        #                                         bonus_max_duration_s, raised
        #                                         to ~60 s so the recovery
        #                                         cycle has time to bring
        #                                         the buggy back to a safe
        #                                         position).
        #   bonus_parking_approach_distance_m    forward distance to drive
        #                                         after the latch fires,
        #                                         BEFORE the LEFT / RIGHT
        #                                         turn begins.  This is
        #                                         the PARKING_APPROACH
        #                                         state — a brief
        #                                         forward-straight step
        #                                         that physically moves
        #                                         the buggy out of the
        #                                         lane before the turn.
        #   bonus_parking_approach_speed         forward speed during
        #                                         PARKING_APPROACH.
        #   bonus_parking_retry_approach_distance_m
        #                                         forward distance to drive
        #                                         straight in RETRY_APPROACH
        #                                         before the next
        #                                         PARKING_TURN begins.
        #                                         This is the "MOVE
        #                                         FORWARD STRAIGHT" step
        #                                         in the spec.
        #   bonus_parking_retry_approach_speed   forward speed during
        #                                         RETRY_APPROACH.
        #   bonus_parking_straighten_dt_s        per-iteration dwell time
        #                                         inside
        #                                         RECOVERY_STRAIGHTEN —
        #                                         how long the buggy
        #                                         applies a small
        #                                         corrective steering
        #                                         before stopping to
        #                                         re-read LiDAR.
        #   bonus_parking_straighten_max_iter   maximum number of
        #                                         iterative
        #                                         straightening
        #                                         iterations before
        #                                         the geometry is
        #                                         declared safe (or
        #                                         we transition to
        #                                         RETRY_APPROACH).
        #   bonus_parking_straighten_correction  magnitude of the
        #                                         small corrective
        #                                         steering applied
        #                                         each iteration
        #                                         (kept small so we
        #                                         don't snap to a
        #                                         large counter-steer).
        #   bonus_parking_straighten_safe_diff_m
        #                                         required
        #                                         FRONT-vs-FRONT-LEFT
        #                                         differential to
        #                                         declare the
        #                                         geometry safe
        #                                         (the robot is
        #                                         roughly aligned
        #                                         with the available
        #                                         forward path).
        #   bonus_parking_emergency_force_stop
        #                                         if True, an
        #                                         EMERGENCY reading
        #                                         inside a recovery
        #                                         sub-state forces
        #                                         immediate zero
        #                                         velocity (in
        #                                         addition to the
        #                                         sub-state machine
        #                                         continuing to
        #                                         advance).  Defaults
        #                                         to True.
        # =====================================================================
        self.declare_parameter('bonus_parking_timeout_s', 60.0)
        self.declare_parameter('bonus_parking_approach_distance_m', 0.20)
        self.declare_parameter('bonus_parking_approach_speed', 0.10)
        self.declare_parameter('bonus_parking_retry_approach_distance_m', 0.30)
        self.declare_parameter('bonus_parking_retry_approach_speed', 0.10)
        self.declare_parameter('bonus_parking_straighten_dt_s', 0.20)
        self.declare_parameter('bonus_parking_straighten_max_iter', 6)
        self.declare_parameter('bonus_parking_straighten_correction', 0.15)
        self.declare_parameter('bonus_parking_straighten_safe_diff_m', 0.15)
        self.declare_parameter('bonus_parking_emergency_force_stop', True)

        # =====================================================================
        # BONUS PARKING — explicit SEARCH / APPROACH / FULL_TURN /
        # STRAIGHTEN / PARK_FORWARD state-machine parameters.
        #
        # The previous bonus controller had only two parking phases
        # (PARKING_LEFT / PARKING_RIGHT) with proportional LiDAR
        # steering.  The new controller follows the user spec:
        # a simple, deterministic, human-like parking maneuver with
        # explicit named phases.  All values are live-tunable via
        # `ros2 param set` and are read every second by
        # `_reload_params()`.
        #
        #   bonus_park_search_speed
        #       Forward speed while in BONUS_SEARCH.
        #   bonus_park_approach_distance_m
        #       Forward distance to drive STRAIGHT after the side
        #       is latched, BEFORE the full turn begins
        #       (BONUS_APPROACH).  ~0.45 m.
        #   bonus_park_approach_speed
        #       Forward speed during BONUS_APPROACH.  0.18 m/s.
        #   bonus_park_full_turn_strength
        #       Magnitude of the side-directed steering command
        #       during BONUS_FULL_TURN.  Per the spec, this is
        #       1.00 (full lock) — the buggy must reach the
        #       full steering command IMMEDIATELY (no slew-
        #       limit, no proportional easing).  The retry
        #       per-attempt scale table (1.00 / 1.00 / 0.95 /
        #       0.95 / 0.90 / 0.90) is applied by
        #       `_run_bonus_tick` against this base value.
        #   bonus_park_full_turn_speed
        #       Forward speed during BONUS_FULL_TURN.  0.14–0.18 m/s.
        #   bonus_park_full_turn_max_duration_s
        #       Maximum time the FULL_TURN phase is allowed to
        #       run.  After this, the FSM transitions to
        #       STRAIGHTEN.  Acts as a safety so the turn does
        #       not run forever.
        #   bonus_park_straighten_duration_s
        #       Time to drive STRAIGHT (steering=0) after the
        #       full turn ends, in BONUS_STRAIGHTEN.  0.50 s.
        #   bonus_park_forward_speed
        #       Forward speed during BONUS_PARK_FORWARD.  0.20 m/s.
        #   bonus_park_forward_distance_m
        #       Forward distance covered in BONUS_PARK_FORWARD
        #       before transitioning to BONUS_PARKED.  The spec
        #       bumped this from 1.20 m to 1.30 m so the buggy
        #       is fully inside the slot before the success log
        #       fires.  "driven_pf >= bonus_park_forward_distance_m"
        #       is the success criterion.
        #
        # ---- Recovery geometry (the new behavior) ----
        #
        #   bonus_park_recovery_stop_duration_s
        #       Time to dwell at zero velocity in the initial
        #       BONUS_RECOVERY_STOP.  0.25 s.
        #   bonus_park_recovery_reverse_speed
        #       Negative speed used in BONUS_RECOVERY_REVERSE.
        #       -0.12 m/s (default; was -0.10 m/s).
        #   bonus_park_recovery_reverse_distance_m
        #       Target reverse distance.  Reverse is terminated
        #       when this distance is reached OR after
        #       `bonus_park_recovery_reverse_max_duration_s`,
        #       whichever comes first.  0.50 m.
        #   bonus_park_recovery_reverse_max_duration_s
        #       Hard upper bound on reverse time.  5.0 s.
        #   bonus_park_recovery_reverse_steering_left
        #       Counter-steer magnitude for LEFT parking while
        #       reversing (axes[3] < 0, physical RIGHT).  -0.55.
        #   bonus_park_recovery_reverse_steering_right
        #       Counter-steer magnitude for RIGHT parking while
        #       reversing (axes[3] > 0, physical LEFT).  +0.55.
        #   bonus_park_recovery_stop2_duration_s
        #       Post-reverse pause.  0.20 s.
        #   bonus_park_recovery_straighten_distance_m
        #       Short forward straighten run after reverse.
        #       0.30 m.  Speed is `bonus_park_approach_speed`
        #       (0.10 m/s in this phase, NOT 0.18 — a slow
        #       alignment nudge).
        #   bonus_park_recovery_stop3_duration_s
        #       Post-straighten pause.  0.10 s.
        #   bonus_park_recovery_forward_distance_m
        #       Forward distance to drive STRAIGHT in the
        #       BONUS_RECOVERY_FORWARD step (to a NEW starting
        #       pose for the retry).  0.65 m.
        #   bonus_park_recovery_forward_speed
        #       Forward speed during BONUS_RECOVERY_FORWARD.
        #       0.18 m/s.
        #   bonus_park_max_attempts
        #       Maximum number of full parking attempts before
        #       the FSM declares BONUS_FAILED.  6 (was 4).
        # =====================================================================
        self.declare_parameter('bonus_park_search_speed', 0.30)
        # APPROACH must be long enough that the buggy physically
        # moves slightly past the FRONT edge of the parking
        # opening before the FULL_TURN begins.  The spec asks
        # for ~0.40-0.50 m; 0.45 m is the default.
        self.declare_parameter('bonus_park_approach_distance_m', 0.45)
        self.declare_parameter('bonus_park_approach_speed', 0.18)
        # Full lock per spec — no slew-limit, no proportional
        # easing.  The retry per-attempt scale table
        # (1.00/1.00/0.95/0.95/0.90/0.90) is applied by
        # `_run_bonus_tick` against this base value.
        self.declare_parameter('bonus_park_full_turn_strength', 1.00)
        self.declare_parameter('bonus_park_full_turn_speed', 0.18)
        # FULL_TURN must be long enough for the buggy to
        # physically swing into the slot.  5.0 s gives a
        # generous turn window for the parked side to be
        # reached even at low speed.
        self.declare_parameter('bonus_park_full_turn_max_duration_s', 5.0)
        self.declare_parameter('bonus_park_straighten_duration_s', 0.50)
        self.declare_parameter('bonus_park_forward_speed', 0.20)
        # PARK_FORWARD is the most important distance in the
        # whole maneuver.  The spec bumped this to 1.30 m so
        # the buggy is fully inside the slot before the
        # success log fires.  "driven_pf >= bonus_park_forward_distance_m"
        # is the success criterion.
        self.declare_parameter('bonus_park_forward_distance_m', 1.30)
        # RETRY approach distance starts a bit larger than
        # the first APPROACH (so each retry physically changes
        # the starting pose) and grows by the increment on
        # each subsequent retry.  (Retained for backward compat —
        # the new recovery uses a single fixed FORWARD distance
        # instead of the incrementing retry table.)
        self.declare_parameter(
            'bonus_park_retry_approach_distance_base_m', 0.45)
        self.declare_parameter(
            'bonus_park_retry_approach_distance_increment_m', 0.10)
        self.declare_parameter('bonus_park_recovery_stop_duration_s', 0.25)
        self.declare_parameter('bonus_park_recovery_reverse_speed', -0.12)
        self.declare_parameter('bonus_park_recovery_reverse_distance_m', 0.50)
        self.declare_parameter(
            'bonus_park_recovery_reverse_max_duration_s', 5.0)
        self.declare_parameter(
            'bonus_park_recovery_reverse_steering_left', -0.55)
        self.declare_parameter(
            'bonus_park_recovery_reverse_steering_right', 0.55)
        self.declare_parameter('bonus_park_recovery_stop2_duration_s', 0.20)
        self.declare_parameter(
            'bonus_park_recovery_straighten_distance_m', 0.30)
        self.declare_parameter('bonus_park_recovery_stop3_duration_s', 0.10)
        self.declare_parameter(
            'bonus_park_recovery_forward_distance_m', 0.65)
        self.declare_parameter(
            'bonus_park_recovery_forward_speed', 0.18)
        self.declare_parameter('bonus_park_recovery_straighten_duration_s', 0.30)
        # Legacy time-based reverse terminator — kept so the
        # old "reverse for 1.5 s" behavior still works if
        # somebody re-toggles the distance-based one.
        self.declare_parameter('bonus_park_recovery_reverse_duration_s', 1.50)
        # Max attempts bumped from 4 to 6 per the spec.
        self.declare_parameter('bonus_park_max_attempts', 6)

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

        # =====================================================================
        # Hospital-mismatch guard (BUG 2 fix)
        # =====================================================================
        # When a HOSPITAL QR is observed by the QR Detector and reported
        # via /target_qr, we record it here.  The hospital-stop decision
        # (`_on_safe_zone_detected`) requires the LAST observed QR to
        # match the active target's assigned hospital.  This is what
        # lets the buggy pass a WRONG hospital without stopping:
        # the close-beam detection fires for the wrong hospital too
        # (it's just a LiDAR feature, not a QR check), but the
        # mismatch with the active target is what protects us.
        self._last_qr_seen = None
        self._last_qr_seen_time = 0.0
        # =====================================================================
        # Bonus-parking override flag (BUG 1 fix)
        # =====================================================================
        # When True, the LiDAR callback is told NOT to set the
        # `obstacle_detected` flag.  The flag is only used by the
        # normal-line-following code path in control_loop, which is
        # already short-circuited during BONUS_PARKING_TEST, but we
        # also clear it here so any other node in the system that
        # reads `obstacle_detected` (or `nearest_dist`) sees a
        # "no obstacle" state during bonus parking.  This prevents
        # the B3RB's built-in obstacle-avoidance node from firing
        # when the buggy is intentionally driving past the parking
        # cones.
        self._bonus_parking_active = False

        # =====================================================================
        # BONUS_PARKING_TEST state (first-hospital debug feature)
        # =====================================================================
        #   _last_edge_msg            cached EdgeVectors for the bonus sub
        #   _bonus_missing_side       current candidate missing side
        #                             (None / "LEFT" / "RIGHT")
        #   _bonus_missing_streak     consecutive frames with the same
        #                             single-vector side
        #   _bonus_clear_side         side currently confirmed clear by LiDAR
        #                             (None / "LEFT" / "RIGHT")
        #   _bonus_clear_streak       consecutive frames LiDAR has confirmed
        #                             the side clear
        #   _bonus_phase              one of the bonus phases
        #                             (BONUS_PARKING_SEARCH /
        #                              BONUS_TURN_LEFT / RIGHT /
        #                              BONUS_PARKING_FORWARD /
        #                              BONUS_PARKED).
        #                             Initial value "SEARCH" is harmless
        #                             because _init_bonus_state() resets
        #                             it on the first BONUS tick.
        #   _bonus_turn_start_dist    travelled dist at TURN-start
        #   _bonus_state_start_time   monotonic time at state entry
        #   _bonus_done               one-shot latch — set to True the
        #                             first time the bonus test runs so
        #                             it never auto-repeats
        #   _bonus_turn_cmd           filtered steering (slew-limited)
        #   _bonus_log_time           throttle for bonus logs
        #   _bonus_left_missing_count independent counter of consecutive
        #                             frames with LEFT as the missing
        #                             side (debug-log field)
        #   _bonus_right_missing_count same for RIGHT
        #   _bonus_latched_side       side we committed to enter
        #                             (None / "LEFT" / "RIGHT")
        #
        # Closed-loop parking-maneuver state (used after the latch
        # fires, while in BONUS_PARKING_TURN / FORWARD / CORRECTION):
        #   _bonus_recovery_count     number of single-tick recovery
        #                             actions (REVERSE/PAUSE/RESUME)
        #                             used so far in this bonus cycle
        #   _bonus_recovery_phase     sub-state of recovery:
        #                             "IDLE" / "REVERSE" / "PAUSE" / "RESUME"
        #   _bonus_recovery_start     monotonic time at the start of
        #                             the current recovery sub-state
        #   _bonus_forward_start_travel
        #                             travel recorded at the start of
        #                             BONUS_PARKING_FORWARD (used to
        #                             count forward distance traveled
        #                             inside the slot)
        #   _bonus_correction_log_time
        #                             throttle for the "PARKING SAFETY"
        #                             one-line log
        #   _bonus_last_safety        last safety verdict logged
        #                             ("CLEAR" / "CAUTION" / "BLOCKED")
        #
        # Retry-aware closed-loop recovery state (separate from the
        # single-tick REVERSE/PAUSE/RESUME above).  Each "attempt"
        # is a full fresh start of the LEFT/RIGHT turn with a
        # smaller turn_scale and a smaller speed_scale:
        #   _bonus_parking_attempt_count
        #                             number of full attempts of the
        #                             TURN maneuver used so far
        #                             (1 = first attempt, 2 = first
        #                             retry, etc.).  Capped by
        #                             bonus_park_max_parking_retries.
        #   _bonus_turn_scale         0..1 scale applied to the turn
        #                             target this attempt.  Reduced
        #                             by bonus_park_turn_reduction_per_retry
        #                             on every retry, clamped to
        #                             bonus_park_min_turn_scale.
        #   _bonus_speed_scale        0..1 scale applied to the
        #                             forward speed this attempt.
        #                             Reduced the same way.
        #   _bonus_recovery_step      higher-level recovery sub-state
        #                             (more granular than the
        #                             _bonus_recovery_phase above).
        #                             Used to drive the spec's
        #                             STOP -> REVERSE -> ALIGN -> RETRY
        #                             cycle:
        #                               "STOP"      — publish zero
        #                                             velocity, log
        #                                             "obstacle
        #                                             detected —
        #                                             STOPPING"
        #                               "REVERSE"   — drive backwards
        #                                             slowly for
        #                                             bonus_park_reverse_*
        #                               "PAUSE"     — brief stop, let
        #                                             the LiDAR catch up
        #                               "ALIGN"     — re-read LiDAR,
        #                                             compute
        #                                             corrective
        #                                             steering
        #                               "RETRY"     — start a new
        #                                             attempt with
        #                                             reduced scale
        #                               "IDLE"      — no recovery
        #                                             in progress
        #   _bonus_recovery_start_travel
        #                             travel distance at the start of
        #                             the current REVERSE step (used
        #                             for distance-based reverse)
        #   _bonus_attempt_start_travel
        #                             travel distance at the start
        #                             of the current attempt (so a
        #                             TURN -> FORWARD transition
        #                             restarts the in-slot distance
        #                             counter on every retry)
        #   _bonus_align_correction   corrective steering computed
        #                             during the ALIGN step (added
        #                             to the turn target so the
        #                             robot actively moves away
        #                             from the cone that blocked
        #                             the previous attempt)
        #   _bonus_recovery_substate_logged
        #                             used to ensure the "PARKING
        #                             RECOVERY" header log is
        #                             emitted exactly once per
        #                             recovery cycle (not on
        #                             every tick)
        # =====================================================================
        self._last_edge_msg = None
        self._bonus_missing_side = None
        self._bonus_missing_streak = 0
        # Per-side INDEPENDENT consecutive-missing counters (used by the
        # spec-format debug log so the operator can see the streak for
        # BOTH sides at the same time, not only for the side the latch
        # has tentatively selected).
        self._bonus_left_missing_count = 0
        self._bonus_right_missing_count = 0
        self._bonus_phase = BonusPhase.SEARCH
        self._bonus_turn_cmd = 0.0
        self._bonus_log_time = 0.0
        self._bonus_latched_side = None
        self._bonus_state_start_time = 0.0
        self._bonus_last_safety = "CLEAR"
        # Retry-aware recovery state (initialised in _init_bonus_state)
        self._bonus_parking_attempt_count = 0
        self._bonus_turn_scale = 1.0
        self._bonus_speed_scale = 1.0
        self._bonus_recovery_step = BonusRecovery.IDLE

        # =====================================================================
        # Explicit SEARCH / APPROACH / FULL_TURN / STRAIGHTEN /
        # PARK_FORWARD state-machine fields
        # =====================================================================
        # The state machine is documented in `_init_bonus_state` and
        # `_run_bonus_tick`.  Each new field has a clear role in
        # the new sub-state machine; nothing here affects the
        # normal line follower or the safe-zone parking.
        #
        #   _bonus_entry_start_travel
        #       travel recorded at the moment the side was
        #       latched (start of BONUS_APPROACH).  The
        #       success criterion at the end of
        #       BONUS_PARK_FORWARD uses this as the
        #       baseline.
        #
        #   _bonus_phase_start_time
        #       monotonic time at the start of the
        #       current `_bonus_phase` (used to time
        #       BONUS_FULL_TURN, BONUS_STRAIGHTEN, and
        #       BONUS_RECOVERY_RETRY).
        #
        #   _bonus_phase_start_travel
        #       travel distance at the start of the
        #       current `_bonus_phase` (used to time
        #       BONUS_APPROACH and BONUS_PARK_FORWARD
        #       by distance).
        #
        #   _bonus_recovery_step
        #       the higher-level recovery sub-state
        #       (IDLE / STOP / REVERSE / STRAIGHTEN /
        #       RETRY).  During recovery the externally
        #       visible `_bonus_phase` stays in the
        #       current active parking phase so the
        #       operator can see which "active" phase is
        #       being recovered.
        #
        #   _bonus_recovery_start_time
        #       monotonic time at the start of the
        #       current recovery sub-state.
        #
        #   _bonus_recovery_start_travel
        #       travel distance at the start of the
        #       current recovery sub-state (used for
        #       BONUS_RECOVERY_RETRY's distance gate).
        #
        #   _bonus_recovery_reverse_distance
        #       dedicated reverse-distance accumulator.
        #       The shared `_travel_dist` only integrates
        #       forward motion (clamped at zero), so we
        #       need a separate counter to terminate
        #       BONUS_RECOVERY_REVERSE by distance.
        #
        #   _bonus_recovery_last_travel_time
        #       monotonic time at the previous tick of
        #       the reverse-distance integrator.
        #
        #   _bonus_parking_attempt_count
        #       1 = first attempt, 2 = first retry,
        #       ...  capped by bonus_park_max_attempts.
        # =====================================================================
        self._bonus_entry_start_travel = 0.0
        self._bonus_phase_start_time = 0.0
        self._bonus_phase_start_travel = 0.0
        self._bonus_recovery_start_time = 0.0
        self._bonus_recovery_start_travel = 0.0
        self._bonus_recovery_reverse_distance = 0.0
        self._bonus_recovery_last_travel_time = 0.0
        self._bonus_parking_attempt_count = 0

        # ---- One-shot log guards (initialised in _init_bonus_state) ----
        self._bonus_latched_logged = False
        self._bonus_approach_logged = False
        self._bonus_full_turn_logged = False
        self._bonus_straighten_logged = False
        self._bonus_park_forward_logged = False
        self._bonus_parked_logged = False
        self._bonus_recovery_stop_logged = False
        self._bonus_recovery_reverse_logged = False
        self._bonus_recovery_straighten_logged = False
        self._bonus_recovery_retry_logged = False

        # ---------------- ROS plumbing ----------------
        self.create_subscription(
            EdgeVectors, '/edge_vectors', self.edge_vectors_callback, QOS_PROFILE_DEFAULT)
        # ---- Parallel, lightweight subscription for the bonus feature ----
        # The existing edge_vectors_callback owns the controller state.
        # This second subscriber only caches the latest message so the
        # bonus branch of control_loop can read vector_count, vector_1.x
        # and vector_1.y without disturbing the existing callback.
        self.create_subscription(
            EdgeVectors, '/edge_vectors', self._bonus_edge_cache, QOS_PROFILE_DEFAULT)

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

    # =====================================================================
    # BONUS_PARKING_TEST — light cache of the latest /edge_vectors message
    # =====================================================================
    def _bonus_edge_cache(self, message):
        """Cache the latest /edge_vectors message for the bonus branch.

        This is a parallel subscriber; the existing edge_vectors_callback
        remains the only owner of the controller state (vectors_available,
        target_turn, target_speed, etc.).  This callback only stores a
        reference to the message and exposes vector_count, vector_1.x and
        vector_1.y to the bonus branch of control_loop.
        """
        self._last_edge_msg = message

    def _transition_mission_state(self, new_state, reason=""):
        """Log and execute a mission state transition."""
        old = self.mission_state
        if old == new_state:
            return
        self.mission_state = new_state

        # When the FSM returns to an idle/accepting state, promote any
        # assignment that was queued while a mission was in progress.
        # NOTE: BONUS_PARKING_TEST is NOT an accepting state — it is a
        # busy state that preserves the next mission in pending_next_*.
        # Promotion happens only when the FSM exits to
        # NORMAL_LINE_FOLLOWING or NAVIGATING_TO_NEXT_TARGET.
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
        self.hospital_qr_max_age_s = float(g('hospital_qr_max_age_s'))

        # BONUS_PARKING_TEST parameters
        self.bonus_enable = bool(g('bonus_enable'))
        self.bonus_speed = float(g('bonus_speed'))
        self.bonus_turn_speed = float(g('bonus_turn_speed'))
        self.bonus_turn_gain = float(g('bonus_turn_gain'))
        self.bonus_straighten_gain = float(g('bonus_straighten_gain'))
        self.bonus_missing_side_frames = int(g('bonus_missing_side_frames'))
        self.bonus_clear_confirm_frames = int(g('bonus_clear_confirm_frames'))
        self.bonus_side_offset_deg = float(g('bonus_side_offset_deg'))
        self.bonus_clear_fov_deg = float(g('bonus_clear_fov_deg'))
        self.bonus_side_clear_m = float(g('bonus_side_clear_m'))
        self.bonus_side_min_beams = int(g('bonus_side_min_beams'))
        self.bonus_side_block_m = float(g('bonus_side_block_m'))
        self.bonus_enter_distance_m = float(g('bonus_enter_distance_m'))
        self.bonus_max_duration_s = float(g('bonus_max_duration_s'))
        self.bonus_turn_slew = float(g('bonus_turn_slew'))
        self.bonus_turn_max = float(g('bonus_turn_max'))
        # Closed-loop maneuver parameters
        self.bonus_park_front_fov_deg = float(g('bonus_park_front_fov_deg'))
        self.bonus_park_frontleft_lo_deg = float(g('bonus_park_frontleft_lo_deg'))
        self.bonus_park_frontleft_hi_deg = float(g('bonus_park_frontleft_hi_deg'))
        self.bonus_park_left_lo_deg = float(g('bonus_park_left_lo_deg'))
        self.bonus_park_left_hi_deg = float(g('bonus_park_left_hi_deg'))
        self.bonus_park_frontright_lo_deg = float(
            g('bonus_park_frontright_lo_deg'))
        self.bonus_park_frontright_hi_deg = float(
            g('bonus_park_frontright_hi_deg'))
        self.bonus_park_right_lo_deg = float(g('bonus_park_right_lo_deg'))
        self.bonus_park_right_hi_deg = float(g('bonus_park_right_hi_deg'))
        self.bonus_park_entry_turn_strength = float(
            g('bonus_park_entry_turn_strength'))
        self.bonus_park_entry_speed = float(g('bonus_park_entry_speed'))
        self.bonus_park_success_clearance_m = float(
            g('bonus_park_success_clearance_m'))
        self.bonus_park_front_emergency_m = float(
            g('bonus_park_front_emergency_m'))
        self.bonus_park_front_blocked_m = float(
            g('bonus_park_front_blocked_m'))
        self.bonus_park_reverse_speed = float(g('bonus_park_reverse_speed'))
        self.bonus_park_reverse_duration_s = float(
            g('bonus_park_reverse_duration_s'))
        self.bonus_park_reverse_min_distance_m = float(
            g('bonus_park_reverse_min_distance_m'))
        self.bonus_park_align_duration_s = float(
            g('bonus_park_align_duration_s'))
        self.bonus_park_stop_dwell_s = float(g('bonus_park_stop_dwell_s'))
        self.bonus_park_max_attempts = int(g('bonus_park_max_attempts'))
        # Backward-compat aliases (kept so existing user-parameter
        # YAML files do not break).  The new names above are the
        # canonical ones.
        self.bonus_park_max_parking_retries = self.bonus_park_max_attempts
        self.bonus_park_enter_distance_m = float(g('bonus_park_enter_distance_m'))

        # Closed-loop bonus parking — new state-machine parameters
        self.bonus_parking_timeout_s = float(g('bonus_parking_timeout_s'))
        self.bonus_parking_approach_distance_m = float(
            g('bonus_parking_approach_distance_m'))
        self.bonus_parking_approach_speed = float(
            g('bonus_parking_approach_speed'))
        self.bonus_parking_retry_approach_distance_m = float(
            g('bonus_parking_retry_approach_distance_m'))
        self.bonus_parking_retry_approach_speed = float(
            g('bonus_parking_retry_approach_speed'))
        self.bonus_parking_straighten_dt_s = float(
            g('bonus_parking_straighten_dt_s'))
        self.bonus_parking_straighten_max_iter = int(
            g('bonus_parking_straighten_max_iter'))
        self.bonus_parking_straighten_correction = float(
            g('bonus_parking_straighten_correction'))
        self.bonus_parking_straighten_safe_diff_m = float(
            g('bonus_parking_straighten_safe_diff_m'))
        self.bonus_parking_emergency_force_stop = bool(
            g('bonus_parking_emergency_force_stop'))

        # ---- Explicit SEARCH / APPROACH / FULL_TURN / STRAIGHTEN
        #      / PARK_FORWARD state-machine parameters ----
        self.bonus_park_search_speed = float(g('bonus_park_search_speed'))
        self.bonus_park_approach_distance_m = float(
            g('bonus_park_approach_distance_m'))
        self.bonus_park_approach_speed = float(
            g('bonus_park_approach_speed'))
        self.bonus_park_full_turn_strength = float(
            g('bonus_park_full_turn_strength'))
        self.bonus_park_full_turn_speed = float(
            g('bonus_park_full_turn_speed'))
        self.bonus_park_full_turn_max_duration_s = float(
            g('bonus_park_full_turn_max_duration_s'))
        self.bonus_park_straighten_duration_s = float(
            g('bonus_park_straighten_duration_s'))
        self.bonus_park_forward_speed = float(
            g('bonus_park_forward_speed'))
        self.bonus_park_forward_distance_m = float(
            g('bonus_park_forward_distance_m'))
        self.bonus_park_retry_approach_distance_base_m = float(
            g('bonus_park_retry_approach_distance_base_m'))
        self.bonus_park_retry_approach_distance_increment_m = float(
            g('bonus_park_retry_approach_distance_increment_m'))
        # New recovery-geometry parameters (per spec rev 2).
        self.bonus_park_recovery_stop_duration_s = float(
            g('bonus_park_recovery_stop_duration_s'))
        self.bonus_park_recovery_reverse_speed = float(
            g('bonus_park_recovery_reverse_speed'))
        self.bonus_park_recovery_reverse_distance_m = float(
            g('bonus_park_recovery_reverse_distance_m'))
        self.bonus_park_recovery_reverse_max_duration_s = float(
            g('bonus_park_recovery_reverse_max_duration_s'))
        self.bonus_park_recovery_reverse_steering_left = float(
            g('bonus_park_recovery_reverse_steering_left'))
        self.bonus_park_recovery_reverse_steering_right = float(
            g('bonus_park_recovery_reverse_steering_right'))
        self.bonus_park_recovery_stop2_duration_s = float(
            g('bonus_park_recovery_stop2_duration_s'))
        self.bonus_park_recovery_straighten_distance_m = float(
            g('bonus_park_recovery_straighten_distance_m'))
        self.bonus_park_recovery_stop3_duration_s = float(
            g('bonus_park_recovery_stop3_duration_s'))
        self.bonus_park_recovery_forward_distance_m = float(
            g('bonus_park_recovery_forward_distance_m'))
        self.bonus_park_recovery_forward_speed = float(
            g('bonus_park_recovery_forward_speed'))
        # Legacy time-based reverse terminator (kept for
        # backward compat with anyone re-toggling the
        # distance-based one).
        self.bonus_park_recovery_reverse_duration_s = float(
            g('bonus_park_recovery_reverse_duration_s'))
        self.bonus_park_recovery_straighten_duration_s = float(
            g('bonus_park_recovery_straighten_duration_s'))

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
          PARKED_IN_SAFE_ZONE / WAITING_FOR_SERVER_ACK / MISSION_COMPLETE
          / BONUS_PARKING_TEST):
          the active mission is IMMUTABLE — the new assignment is queued
          as pending_next_* and promoted only after the current mission
          finishes.  In particular, while the bonus state is running,
          ANY incoming /target_type for the next mission is preserved
          in pending_next_type and never lost.
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
        mission completes.  In particular, while the bonus state is
        running, ANY incoming /target_qr for the next mission is
        preserved in pending_next_qr and never lost.

        Backward compatibility: with an older QR Detector that does not
        publish /target_type, the QR is kept pending and the mission is
        activated with "UNKNOWN" after target_type_wait_timeout (legacy
        parking profile).
        """
        if not msg.data or not msg.data.strip():
            return
        qr = msg.data.strip()

        # BUG 2 fix: record the LAST QR observed (regardless of
        # whether it matches the active target).  This is the
        # ground truth for the hospital-mismatch guard:
        # `_on_safe_zone_detected` consults `_last_qr_seen` to
        # decide whether the close-beam cluster corresponds to
        # the assigned hospital (in which case the buggy should
        # stop and deliver) or a different/wrong hospital (in
        # which case the buggy should keep moving).
        prev_qr = self._last_qr_seen
        self._last_qr_seen = qr
        self._last_qr_seen_time = time.time()
        # When we are on a HOSPITAL mission and a wrong hospital
        # QR is observed, emit the spec's MISMATCH log so the
        # operator can see why we are not stopping.
        if (self._mission_in_progress()
                and self.active_target_type == "HOSPITAL"
                and qr.upper().startswith("HOSPITAL_")
                and self.active_target_qr
                and qr != self.active_target_qr
                and qr != prev_qr):
            self.get_logger().info(
                "========================================\n"
                "HOSPITAL QR DETECTED: " + qr + "\n"
                "ASSIGNED HOSPITAL: " + self.active_target_qr + "\n"
                "MATCH: NO\n"
                "HOSPITAL MISMATCH\n"
                "Continuing navigation — NO STOP\n"
                "========================================")
        elif (self._mission_in_progress()
                and self.active_target_type == "HOSPITAL"
                and qr == self.active_target_qr
                and qr != prev_qr):
            self.get_logger().info(
                "========================================\n"
                "HOSPITAL QR DETECTED: " + qr + "\n"
                "ASSIGNED HOSPITAL: " + self.active_target_qr + "\n"
                "MATCH: YES\n"
                "========================================")

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
        be queued instead of applied.

        BONUS_PARKING_TEST is treated as busy: any /target_type or
        /target_qr that arrives for the next mission while we are
        running the bonus test is queued in pending_next_* and
        preserved until the bonus state exits.  This is the key
        invariant that keeps the next mission from being lost.
        """
        return self.mission_state in (
            MissionState.WAITING_FOR_SAFE_ZONE,
            MissionState.ENTERING_SAFE_ZONE,
            MissionState.PARKED_IN_SAFE_ZONE,
            MissionState.WAITING_FOR_SERVER_ACK,
            MissionState.BONUS_PARKING_TEST,
            MissionState.MISSION_COMPLETE,
        )

    def _clear_pending_assembly(self):
        """Reset the incoming-assignment assembly slots."""
        self.pending_target_type = None
        self.pending_target_qr = None
        self.pending_target_qr_time = None
        self.mission_ready = False

    def _store_pending_next_from_assignment(self, assignment):
        """Mirror the assignment from /mission/available into pending_next_*.

        This is the safety net that guarantees the next mission is never
        lost, even if the QR Detector does NOT subsequently publish
        /target_type and /target_qr for the same assignment.  Called
        from mission_available_callback the moment the server message
        arrives.

        Type is inferred from the QR prefix:
            "PATIENT_x"  -> "PATIENT"
            "HOSPITAL_x" -> "HOSPITAL"
            anything else -> "UNKNOWN"

        Existing pending_next_* values are preserved (we only fill
        missing slots); if a more specific value arrives later via
        /target_type or /target_qr, the existing callback logic will
        upgrade or replace it.
        """
        if not assignment:
            return
        ttype = _infer_target_type_from_qr(assignment)
        if self.pending_next_qr is None:
            self.pending_next_qr = assignment
            self.pending_next_qr_time = time.time()
        if self.pending_next_type is None:
            self.pending_next_type = ttype
        self.get_logger().info(
            f"Mission assignment mirrored to pending_next: "
            f"type={self.pending_next_type} qr={self.pending_next_qr} "
            f"(from /mission/available payload '{assignment}')")

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

        pending_next_* is NEVER touched here — the next assignment
        (preserved from /mission/available or queued by the
        /target_type and /target_qr callbacks while a mission was
        busy) survives across active-mission boundaries.
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

        # --- BONUS_PARKING_TEST: /mission/turn is ignored.  The bonus
        #     state owns the drive commands end-to-end; an external
        #     LEFT/RIGHT/STRAIGHT would only disturb the search and
        #     turn logic.  Logged for transparency. ---
        if self.mission_state == MissionState.BONUS_PARKING_TEST:
            self.get_logger().info(
                f"Mission Topic ignored in BONUS_PARKING_TEST: {mission}")
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

        Valid payloads:
          * "PATIENT_x"  -> a normal patient delivery assignment
          * "HOSPITAL_x" -> a normal hospital delivery assignment
          * "BONUS"      -> a special INDEPENDENT parking task.
                            Accepted UNCONDITIONALLY regardless of the
                            current FSM state, the previous hospital
                            visited, the active target type, or any
                            one-shot latch.  Triggers BONUS_PARKING_TEST
                            directly.  Does NOT require /target_type or
                            /target_qr.

        For non-BONUS payloads the controller validates the identifier
        and immediately transitions:
            MISSION_COMPLETE / PARKED_IN_SAFE_ZONE
                ↓
            NAVIGATING_TO_NEXT_TARGET

        Line following resumes immediately — /mission/turn is NOT
        required to start moving.  The object recognition node remains
        the only publisher of /mission/turn; it only updates the turn
        direction later while navigating.  No additional resume message
        is needed.  Event-driven: no polling, no timers.

        BONUS is an independent command.  There are NO conditions on
        which hospital was completed, on the active target type or
        target QR, or on a one-shot latch.  The string "BONUS" itself
        is never mirrored into pending_next_* (it is not a real
        target).  Any real PATIENT_x / HOSPITAL_x assignment that
        arrives LATER via /mission/available, /target_type, or
        /target_qr is preserved in pending_next_* by the existing
        callbacks (because BONUS_PARKING_TEST is treated as a busy
        state by _mission_in_progress()) and activated when the bonus
        state exits.

        MISSION PRESERVATION: _mission_in_progress() returns True for
        BONUS_PARKING_TEST, so any /target_type or /target_qr that
        arrives for the next mission while we are running the bonus
        is queued in pending_next_* and never lost.  When the bonus
        state later exits to NAVIGATING_TO_NEXT_TARGET,
        _promote_pending_next() activates the preserved assignment.
        """
        if not msg.data or not msg.data.strip():
            return

        assignment = msg.data.strip()

        # =====================================================================
        # BONUS TASK — accepted UNCONDITIONALLY
        # =====================================================================
        # "BONUS" is a special Municipality-Server signal that asks the
        # controller to immediately start an independent parking task.
        # It is NOT a normal target mission, so it must never be stored
        # in pending_next_* and must not activate a WAITING_FOR_SAFE_ZONE
        # cycle.
        #
        # There is NO check on:
        #   - which hospital was previously visited
        #   - the active target type or target QR
        #   - the current FSM state
        #   - any one-shot "_bonus_done" latch
        #
        # The ONLY check is the master switch bonus_enable.  If the
        # caller wants to disable the bonus task entirely, they set
        # bonus_enable=false via ros2 param.
        # =====================================================================
        if assignment.upper() == "BONUS":
            if not self.bonus_enable:
                self.get_logger().warn(
                    "BONUS TASK RECEIVED but bonus_enable=false -> ignoring")
                return

            self.get_logger().info(
                "========================================\n"
                "BONUS TASK RECEIVED\n"
                "Entering BONUS_PARKING mode\n"
                "Normal obstacle avoidance: DISABLED\n"
                "Normal line following override: ACTIVE\n"
                "========================================")

            # BUG 1 fix: mark bonus parking as active so the LiDAR
            # callback stops feeding the normal obstacle-avoidance
            # pipeline (and any external node that watches
            # `obstacle_detected` / `nearest_dist` will see a
            # "no obstacle" state while we are parking).
            self._bonus_parking_active = True
            # Reset per-cycle bonus state so a fresh search starts.
            self._init_bonus_state()
            # Clear the active mission context (whatever it was — the
            # user explicitly required unconditional acceptance).
            # The mission FSM will be in BONUS_PARKING_TEST from this
            # point on, so it does not matter what active_* was.
            self._complete_active_mission("BONUS task accepted unconditionally")
            # Enter the bonus state.  This transitions
            # mission_state -> BONUS_PARKING_TEST and triggers
            # _promote_pending_next() (no-op here because we are
            # leaving an accepting state via the explicit transition).
            self._transition_mission_state(
                MissionState.BONUS_PARKING_TEST,
                "Server requested BONUS task (accepted unconditionally)")
            return

        # =====================================================================
        # Normal PATIENT_x / HOSPITAL_x assignment
        # =====================================================================
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
            # ---- Preserve the next assignment in pending_next_* ----
            # Done FIRST, before clearing the active mission, so the
            # assignment is never lost.
            self._store_pending_next_from_assignment(assignment)
            # ---- Now release the just-completed active mission ----
            self._complete_active_mission("New mission assigned by QR Detector")
            self._transition_mission_state(
                MissionState.NAVIGATING_TO_NEXT_TARGET,
                f"New mission assigned by QR Detector ({assignment})")
            return

        # ---- /mission/available arrived while a different mission is
        #      active (e.g. WAITING_FOR_SAFE_ZONE, BONUS_PARKING_TEST,
        #      ENTERING_SAFE_ZONE).  We still preserve the assignment
        #      in pending_next_* so it survives, and log that the
        #      active mission takes precedence for now. ----
        if self._mission_in_progress():
            if self.pending_next_qr is None:
                self._store_pending_next_from_assignment(assignment)
                self.get_logger().info(
                    f"/mission/available ({assignment}) received while "
                    f"{self.mission_state} active — preserved as "
                    "pending_next.")
            else:
                self.get_logger().info(
                    f"/mission/available ({assignment}) received while "
                    f"{self.mission_state} active and pending_next already "
                    f"set ({self.pending_next_qr}) — ignored to avoid "
                    "clobbering.")
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

    def _clamp_to_lane(self, lane_center, left_edge, right_edge):
        """Hard lane-keeping guard.

        Constrains the aim point to stay strictly inside the detected lane
        boundaries [left_edge, right_edge], keeping at least
        turn_edge_margin_px from each edge.  If the lane is too narrow to fit
        the margins, the aim is centered.  This makes it impossible for the
        controller to steer the robot out of the lane, even under junction
        bias, single-edge tracking or noise.
        """
        margin = self.turn_edge_margin_px
        if right_edge - left_edge <= 2.0 * margin:
            return 0.5 * (left_edge + right_edge)
        return max(left_edge + margin, min(right_edge - margin, lane_center))

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
                                # Aim point is left of center; apply inward bias for left turn
                                lane_center = left_x + 0.50 * self.learned_lane_width
                            else:
                                # Aim point is right of center; apply inward bias for right turn
                                lane_center = right_x - 0.50 * self.learned_lane_width
                            # Apply curvature-based bias: shift away from outer lane edge during turns
                            if abs(current_heading) > math.radians(5):
                                # Positive heading => left turn, bias rightwards; negative => right turn, bias leftwards
                                bias = -math.copysign(0.05 * lane_width, current_heading)
                                lane_center = self._clamped_offset(lane_center + bias, lane_width)
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

                        lane_center = self._clamp_to_lane(lane_center, left_x, right_x)
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

                    # Hard lane-keeping guard: aim point can never leave the lane.
                    lane_center = self._clamp_to_lane(lane_center, left_x, right_x)
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

                # Hard lane-keeping guard: aim point can never leave the lane.
                lane_center = self._clamp_to_lane(lane_center, left_x, right_x)
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

            # Hard lane-keeping guard: with only one edge visible, the aim
            # point is constrained to the reconstructed lane so the robot can
            # never steer outside it.  When the seen edge is the LEFT side the
            # lane spans [aim, aim + lane_width]; when RIGHT it spans
            # [aim - lane_width, aim].
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
    def _update_zone_cache(self, ranges, n, angle_min, angle_increment):
        """Recompute the safe-zone / hospital LiDAR cache from a scan.

        Extracted as a helper so it can be called from both the
        normal-line-following path of `lidar_callback` and from the
        BONUS_PARKING override branch (which still wants the cache
        fresh for the parking controller's own use).
        """
        p_min_deg = self.zone_fov_min_deg
        p_max_deg = self.zone_fov_max_deg
        i_p_min = int(round((math.radians(p_min_deg) - angle_min) / angle_increment))
        i_p_max = int(round((math.radians(p_max_deg) - angle_min) / angle_increment))
        i_p_min = max(0, min(n - 1, i_p_min))
        i_p_max = max(0, min(n - 1, i_p_max))

        valid_distances = []
        sector_data = []
        close_count = 0
        min_dist = float('inf')
        for k in range(i_p_min, i_p_max + 1):
            idx = k % n
            r = ranges[idx]
            angle_deg = math.degrees(angle_min + idx * angle_increment)
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

    def lidar_callback(self, msg):
        if not self.obstacle_enable:
            self.obstacle_detected = False
            return
        ranges = msg.ranges
        n = len(ranges)
        if n == 0 or msg.angle_increment == 0.0:
            return

        # BUG 1 fix: during BONUS_PARKING mode, the normal
        # obstacle-avoidance pipeline is OVERRIDDEN.  We still
        # update the LiDAR cache (so the parking controller can
        # use it for its own safety checks), but we explicitly
        # do NOT set `obstacle_detected = True`.  This prevents
        # the B3RB's built-in obstacle-avoidance node (or any
        # other node reading `obstacle_detected` / `nearest_dist`)
        # from triggering a stop while the parking controller is
        # intentionally driving past the parking cones.  The
        # parking controller has its own dedicated close-range
        # safety (parking_critical_m, parking_emergency_m) so we
        # do NOT lose any safety here — we only disable the
        # normal-line-following obstacle avoidance.
        if self._bonus_parking_active:
            self.obstacle_detected = False
            # Still update the LiDAR cache so the parking
            # controller's sector read works.
            self._last_ranges = ranges
            self._last_range_count = n
            self._last_angle_min = msg.angle_min
            self._last_angle_increment = msg.angle_increment
            self._log_lidar_orientation()
            # IMPORTANT: still update the safe-zone / wall
            # caches (so a hospital stop could fire on a wrong
            # hospital — but BUG 2 also fixes that by checking
            # the QR match).  This way the bonus controller has
            # fresh LiDAR data without us setting
            # `obstacle_detected`.
            self._update_zone_cache(ranges, n, msg.angle_min, msg.angle_increment)
            self._update_wall_geometry(ranges, n, msg.angle_min,
                                       msg.angle_increment,
                                       getattr(msg, 'range_max', float('inf')))
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
        self._update_zone_cache(ranges, n, msg.angle_min, msg.angle_increment)

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
        # COMMAND-PRIORITY ARBITRATION (BUG 1 + BUG 2 fix)
        # =================================================================
        # This controller is the ONLY producer of /cerebri/in/joy.
        # Other nodes (the B3RB's built-in obstacle-avoidance
        # node, the QR Detector's hospital-stop node, etc.) may
        # observe the state we publish (obstacle_detected,
        # nearest_dist, active_target_qr, _last_qr_seen, ...) but
        # they MUST NOT override our commands.  The priority
        # dispatch below is what enforces that:
        #
        #   1. EMERGENCY SAFETY STOP
        #      -> The bonus parking controller publishes zero
        #         velocity when EMERGENCY is detected (see
        #         _bonus_safety_verdict / _run_bonus_tick).  The
        #         safe-zone / hospital stop is also a hard stop,
        #         published via publish_drive_cmd(0, 0).
        #   2. BONUS PARKING CONTROLLER (when BONUS_PARKING_TEST
        #      is active)
        #      -> The dispatch returns early at the BONUS branch
        #         below; the bonus controller is the sole owner of
        #         /cerebri/in/joy during bonus.  _bonus_parking_active
        #         also forces obstacle_detected = False in
        #         lidar_callback so the B3RB's obstacle-avoidance
        #         node sees a "no obstacle" state and does not
        #         fight us.
        #   3. HOSPITAL DELIVERY STOP (only after CORRECT QR +
        #      VALID ZONE)
        #      -> _on_safe_zone_detected now verifies the last
        #         observed /target_qr matches the active
        #         active_target_qr before committing to
        #         ENTERING_SAFE_ZONE.  A wrong hospital QR is
        #         treated as "do not stop" and the buggy keeps
        #         moving.
        #   4. NORMAL OBSTACLE AVOIDANCE
        #      -> Lines 2300+ (the obstacle_detected branch in
        #         the driving logic).
        #   5. NORMAL LINE FOLLOWING
        #      -> Default.
        # =================================================================

        # BUG 1 fix: as long as the FSM is no longer in the bonus
        # parking state, the bonus-parking override is OFF.  This
        # is a safety net for the (rare) case where a transition
        # out of BONUS_PARKING_TEST happened without explicitly
        # clearing `_bonus_parking_active`.  It guarantees the
        # normal obstacle-avoidance pipeline is re-enabled the
        # moment we leave the bonus state.
        if (self.mission_state != MissionState.BONUS_PARKING_TEST
                and self._bonus_parking_active):
            self._bonus_parking_active = False

        # =================================================================
        # Mission assembly watchdog (atomic /target_type + /target_qr)
        # =================================================================
        # NOTE: this branch must NOT run while in BONUS_PARKING_TEST — the
        # bonus state preserves pending_next_* and the next-mission
        # activation is deferred until the bonus state exits to
        # NAVIGATING_TO_NEXT_TARGET (which calls _promote_pending_next
        # via _transition_mission_state).  Activating here would race
        # with the bonus state.
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
        # BONUS_PARKING_TEST — runs to completion before any other
        # control path.  Bypasses the normal driving logic, the safe-
        # zone detector and the parking detector.  When this branch
        # returns, the FSM has already transitioned out of the bonus
        # state (success or safety abort), so the rest of control_loop
        # continues normally.
        # =================================================================
        if self.mission_state == MissionState.BONUS_PARKING_TEST:
            self._update_travel_distance(now)
            self._run_bonus_tick(now)
            return

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

        # BUG 2 fix: hospital-mismatch guard.  The LiDAR close-beam
        # density fires for ANY building the buggy approaches, not
        # just the assigned one.  Before we commit to ENTERING_SAFE_ZONE
        # (which is a hard stop), verify:
        #   1. The active mission is a HOSPITAL mission.
        #   2. The LAST observed /target_qr matches the assigned
        #      hospital, OR is too old to be a reliable
        #      wrong-hospital indicator.
        # If the QR detector is currently reporting a WRONG hospital
        # (and the read is fresh), we refuse to commit to a stop
        # and the buggy keeps moving.  When the QR is too old to
        # be a reliable wrong-hospital signal, we trust the LiDAR
        # close-beam cluster and commit to the stop.
        if (self.active_target_type == "HOSPITAL"
                and self.active_target_qr):
            last_qr = self._last_qr_seen
            qr_age = (time.time() - self._last_qr_seen_time
                      if self._last_qr_seen_time > 0.0 else float('inf'))
            max_age = self.hospital_qr_max_age_s
            # 1) No QR observed yet -> trust the LiDAR, proceed
            #    (this is the legacy behavior; the QR detector may
            #    just not have published anything).
            if last_qr is None:
                pass  # fall through to commit
            # 2) Last QR is for a NON-hospital identifier (e.g. a
            #    PATIENT QR seen in passing) -> trust the LiDAR.
            elif not last_qr.upper().startswith("HOSPITAL_"):
                pass
            # 3) Last QR is for a HOSPITAL, but a WRONG one, and
            #    the read is FRESH -> REFUSE the stop.
            elif (last_qr != self.active_target_qr
                  and qr_age <= max_age):
                self.get_logger().info(
                    "========================================\n"
                    "SAFE ZONE DETECTED — BUT HOSPITAL QR MISMATCH\n"
                    f"Assigned hospital : {self.active_target_qr}\n"
                    f"Last QR seen      : {last_qr} "
                    f"(age {qr_age:.1f}s)\n"
                    "Refusing to enter safe zone — continuing navigation.\n"
                    "========================================")
                # Reset the streak so we don't immediately re-fire.
                return
            # 4) Last QR is for a HOSPITAL, but a WRONG one, and
            #    the read is STALE -> trust the LiDAR (the
            #    mismatch is too old to act on).
            elif (last_qr != self.active_target_qr
                  and qr_age > max_age):
                self.get_logger().info(
                    "========================================\n"
                    "SAFE ZONE DETECTED — HOSPITAL QR STALE\n"
                    f"Assigned hospital : {self.active_target_qr}\n"
                    f"Last QR seen      : {last_qr} "
                    f"(age {qr_age:.1f}s)\n"
                    "Continuing because QR observation is too old to be a "
                    "reliable wrong-hospital indicator.\n"
                    "========================================")
            # 5) Last QR matches the assigned hospital -> proceed.

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
            f"Assigned QR   : {self.active_target_qr}\n"
            f"Last QR seen  : {self._last_qr_seen}\n"
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

    # =====================================================================
    # BONUS_PARKING_TEST — first-hospital debug state
    # =====================================================================
    #
    #   In this state the line-follower / safe-zone / wall-parallel
    #   parking logic is BYPASSED.  Only the following is active:
    #
    #     - /edge_vectors   (parallel subscription _bonus_edge_cache)
    #     - /scan           (used by _bonus_scan_side_clear)
    #     - /odom           (used for the bonus forward-distance enter
    #                        measurement, via _update_travel_distance)
    #
    #   The state machine has two phases:
    #
    #     SEARCH  :  move slowly forward, watch for one lane edge to
    #                disappear consistently for bonus_missing_side_frames
    #                consecutive frames.  When a side becomes a
    #                candidate, also require the same side to be
    #                LiDAR-clear for bonus_clear_confirm_frames
    #                consecutive frames.  Either condition failing
    #                resets the corresponding streak.
    #
    #     TURN    :  once both streaks are satisfied, freeze the
    #                candidate side and slowly steer toward it while
    #                continuing forward (bonus_turn_speed) for
    #                bonus_enter_distance_m of travel, OR until a
    #                safety abort fires.  Steering is slew-limited so
    #                the command is never aggressive.
    #
    #   Exit paths:
    #
    #     - SUCCESS  (forward distance covered)    -> NAVIGATING_TO_NEXT_TARGET
    #     - SAFETY   (opposite side disappears,
    #                 or intended side blocked, or
    #                 bonus_max_duration_s exceeded) -> MISSION_COMPLETE
    #
    #   MISSION PRESERVATION:
    #     _mission_in_progress() returns True for BONUS_PARKING_TEST,
    #     so any /target_type or /target_qr that arrives for the next
    #     mission while we are running the bonus is queued in
    #     pending_next_* and never lost.  On SUCCESS the FSM
    #     transitions to NAVIGATING_TO_NEXT_TARGET, which triggers
    #     _promote_pending_next() and activates the preserved
    #     assignment.  On SAFETY ABORT the FSM goes to
    #     MISSION_COMPLETE; pending_next_* is preserved and will be
    #     promoted on the next /mission/available cycle.
    # =====================================================================

    def _init_bonus_state(self):
        """Reset all per-cycle bonus bookkeeping on entry.

        Explicit state machine:

            BONUS_SEARCH
                drive forward slowly; require a stable
                single-side indication for
                `bonus_missing_side_frames` consecutive
                ticks before latching.  On latch, the
                latched side is recorded and the FSM
                transitions to BONUS_APPROACH.

            BONUS_APPROACH
                drive STRAIGHT forward for
                `bonus_park_approach_distance_m`.  Steering
                is 0.

            BONUS_FULL_TURN
                drive forward slowly while turning STRONGLY
                toward the latched side for
                `bonus_park_full_turn_max_duration_s` (or
                until FRONT is critically close).  No
                lane-vector steering.  Steering is
                `bonus_park_full_turn_strength` in the
                latched-side direction.

            BONUS_STRAIGHTEN
                steering = 0; continue forward for
                `bonus_park_straighten_duration_s`.

            BONUS_PARK_FORWARD
                drive STRAIGHT forward (steering = 0) for
                `bonus_park_forward_distance_m`.  Reaching
                this distance AND FRONT being clear
                transitions to BONUS_PARKED.

            BONUS_PARKED  (terminal)
                linear.x = 0, steering = 0.  Latched side
                and final progress are logged.

            BONUS_FAILED  (terminal)
                linear.x = 0, steering = 0.  Reason is
                logged.

        Recovery sub-state machine (entered from any active
        parking phase on EMERGENCY/BLOCKED/CRITICAL):

            BONUS_RECOVERY_STOP
                linear.x = 0 for
                `bonus_park_recovery_stop_duration_s`.
            BONUS_RECOVERY_REVERSE
                linear.x = `bonus_park_recovery_reverse_speed`
                (negative) for
                `bonus_park_recovery_reverse_duration_s`.
            BONUS_RECOVERY_STRAIGHTEN
                stop, re-read LiDAR, briefly apply a small
                corrective steering nudge for
                `bonus_park_recovery_straighten_duration_s`.
            BONUS_RECOVERY_RETRY
                drive STRAIGHT forward (steering = 0) for
                `bonus_park_retry_approach_distance_base_m`
                + (attempt-2) *
                `bonus_park_retry_approach_distance_increment_m`,
                then transition back to BONUS_FULL_TURN
                on the SAME latched side.

        Each retry physically changes the starting pose
        because the per-retry approach distance is increased.
        The retry ALWAYS targets the SAME latched side
        (no re-decision of LEFT / RIGHT after a failed
        attempt).
        """
        # ---- Perception bookkeeping ----
        self._bonus_missing_side = None
        self._bonus_missing_streak = 0
        self._bonus_left_missing_count = 0
        self._bonus_right_missing_count = 0
        self._bonus_last_kind = "ZERO"
        self._bonus_last_visible_side = "NONE"
        self._bonus_latched_side = None

        # ---- Steering / state-machine bookkeeping ----
        # `_bonus_turn_cmd` is in the JOYSTICK sign convention
        # (positive = physical LEFT, negative = physical
        # RIGHT).  It is published directly via
        # publish_drive_cmd(speed, _bonus_turn_cmd) without
        # any further sign multiplication.
        self._bonus_phase = BonusPhase.SEARCH
        self._bonus_turn_cmd = 0.0
        self._bonus_log_time = 0.0
        self._bonus_state_start_time = time.time()
        self._bonus_last_safety = "CLEAR"

        # ---- Phase / distance / time anchors ----
        self._bonus_entry_start_travel = 0.0
        self._bonus_phase_start_time = 0.0
        self._bonus_phase_start_travel = 0.0

        # ---- Recovery sub-state machine ----
        self._bonus_recovery_step = BonusRecovery.IDLE
        self._bonus_recovery_start_time = 0.0
        self._bonus_recovery_start_travel = 0.0
        self._bonus_recovery_reverse_distance = 0.0
        self._bonus_recovery_last_travel_time = 0.0

        # ---- Per-attempt retry counter ----
        # 1 = first attempt, 2 = first retry, ... capped by
        # bonus_park_max_attempts (default 4).
        self._bonus_parking_attempt_count = 0

        # ---- One-shot log guards ----
        self._bonus_latched_logged = False
        self._bonus_approach_logged = False
        self._bonus_full_turn_logged = False
        self._bonus_straighten_logged = False
        self._bonus_park_forward_logged = False
        self._bonus_parked_logged = False
        self._bonus_recovery_stop_logged = False
        self._bonus_recovery_reverse_logged = False
        self._bonus_recovery_straighten_logged = False
        self._bonus_recovery_retry_logged = False
        # ---- Backward-compat fields (set for older callers) ----
        self._bonus_recovery_start = 0.0
        self._bonus_recovery_substate_logged = False
        self._bonus_retry_logged = False
        self._bonus_recovery_realign_logged = False
        self._bonus_turn_scale = 1.0
        self._bonus_speed_scale = 1.0
        self._bonus_done = False

    def _classify_edge_vector(self):
        """Read the latest /edge_vectors and return one of:
            ("TWO",   None)
            ("ONE",   "LEFT"  | "RIGHT" | "CENTER")
            ("ZERO",  None)

        The LEFT/RIGHT/CENTER classification uses the same band rule as
        the existing single-vector branch of edge_vectors_callback
        (mean-x vs image center with single_vector_side_margin).
        """
        msg = self._last_edge_msg
        if msg is None:
            return ("ZERO", None)
        count = int(getattr(msg, "vector_count", 0))
        img_w = float(getattr(msg, "image_width", 0) or 0)
        if img_w <= 0:
            return ("ZERO", None)
        img_center = img_w / 2.0
        band = self.side_margin * img_center

        if count >= 2:
            return ("TWO", None)
        if count == 1:
            v = msg.vector_1
            try:
                mx = (v[0].x + v[1].x) / 2.0
            except Exception:
                return ("ZERO", None)
            if mx < img_center - band:
                return ("ONE", "LEFT")
            if mx > img_center + band:
                return ("ONE", "RIGHT")
            return ("ONE", "CENTER")
        return ("ZERO", None)

    def _bonus_scan_side_clear(self, side):
        """Return (clear, min_dist, n_beams) for one side of the robot.

        The angular sector is computed FROM THE SCAN METADATA every
        call (angle_min, angle_increment) — no hard-coded beam numbers.
        Given your observed convention
            beam 180 -> +0.5°  (FRONT)
            beam 210 -> +30.6° (FRONT-RIGHT)
            beam 150 -> -29.6° (FRONT-LEFT)
        we expect positive angles to map to the RIGHT side of the
        robot and negative angles to map to the LEFT side, but the
        actual physical convention can be verified against the
        existing _log_lidar_orientation() output.  This function uses
        the raw scan angles, not a beam index, so the orientation
        convention is a free parameter set by bonus_side_offset_deg
        and bonus_clear_fov_deg.

        side == "RIGHT"  -> angular sector
            [ bonus_side_offset_deg , bonus_side_offset_deg + bonus_clear_fov_deg ]
        side == "LEFT"   -> angular sector
            [ -(bonus_side_offset_deg + bonus_clear_fov_deg) , -bonus_side_offset_deg ]

        A side is "clear" when:
            * n_beams >= bonus_side_min_beams   (enough finite samples)
            * min_dist >= bonus_side_clear_m    (nothing too close)
        """
        ranges = self._last_ranges
        n = self._last_range_count
        if n == 0 or self._last_angle_increment == 0.0:
            return (False, float('inf'), 0)

        if side == "RIGHT":
            lo = self.bonus_side_offset_deg
            hi = self.bonus_side_offset_deg + self.bonus_clear_fov_deg
        else:
            lo = -(self.bonus_side_offset_deg + self.bonus_clear_fov_deg)
            hi = -self.bonus_side_offset_deg

        lo_rad = math.radians(lo)
        hi_rad = math.radians(hi)
        i_lo = int(round((lo_rad - self._last_angle_min) / self._last_angle_increment))
        i_hi = int(round((hi_rad - self._last_angle_min) / self._last_angle_increment))
        i_lo = max(0, min(n - 1, i_lo))
        i_hi = max(0, min(n - 1, i_hi))
        if i_hi < i_lo:
            i_lo, i_hi = i_hi, i_lo

        min_dist = float('inf')
        n_beams = 0
        for i in range(i_lo, i_hi + 1):
            r = ranges[i % n]
            if not math.isfinite(r) or r <= 0.05:
                continue
            n_beams += 1
            if r < min_dist:
                min_dist = r

        is_clear = (n_beams >= self.bonus_side_min_beams
                    and min_dist >= self.bonus_side_clear_m)
        return (is_clear, min_dist, n_beams)

    def _bonus_sector_min_dist(self, lo_deg, hi_deg):
        """Return (min_dist, n_finite, n_total) over the angular range
        [lo_deg, hi_deg] (in degrees).  The sector is derived from the
        scan metadata (angle_min / angle_increment) every call — no
        hard-coded beam indices.  Infinite ranges are ignored when
        computing the minimum, but they still count towards n_total
        (so a sector full of inf reads as n_finite == 0, n_total >=
        n_beams; a caller can treat n_finite == 0 as "no obstacle
        within the finite-beam range of the lidar" — effectively
        "open" in the parking-opening sense).

        If the LiDAR cache is empty / invalid, returns
        (inf, 0, 0).
        """
        ranges = self._last_ranges
        n = self._last_range_count
        if n == 0 or self._last_angle_increment == 0.0:
            return (float('inf'), 0, 0)
        # If the range is inverted, swap so lo <= hi.
        if lo_deg > hi_deg:
            lo_deg, hi_deg = hi_deg, lo_deg
        lo_rad = math.radians(lo_deg)
        hi_rad = math.radians(hi_deg)
        i_lo = int(round((lo_rad - self._last_angle_min) / self._last_angle_increment))
        i_hi = int(round((hi_rad - self._last_angle_min) / self._last_angle_increment))
        i_lo = max(0, min(n - 1, i_lo))
        i_hi = max(0, min(n - 1, i_hi))
        if i_hi < i_lo:
            i_lo, i_hi = i_hi, i_lo

        min_dist = float('inf')
        n_finite = 0
        n_total = i_hi - i_lo + 1
        for i in range(i_lo, i_hi + 1):
            r = ranges[i % n]
            if not math.isfinite(r):
                # NaN or inf — ignore for the minimum, do not count
                # as a finite reading.
                continue
            if r <= 0.05:
                # Ground-bounce artifact — ignore.
                continue
            n_finite += 1
            if r < min_dist:
                min_dist = r
        return (min_dist, n_finite, n_total)

    def _bonus_parking_sectors(self):
        """Compute the five parking-safety LiDAR sectors.

        The sectors are centred on the robot's own forward axis
        (positive = the robot's RIGHT, negative = the robot's LEFT,
        regardless of the sign convention used by the joystick),
        because that matches the physical meaning of the sectors.
        The user-supplied parameters bonus_park_front_fov_deg,
        bonus_park_frontleft_*_deg, bonus_park_left_*_deg,
        bonus_park_frontright_*_deg and bonus_park_right_*_deg give
        the bounds in DEGREES with the natural sign convention
        (negative on the left, positive on the right).

        Returns a dict with five keys:
            {
                "front":       (min_dist, n_finite, n_total),
                "front_left":  (min_dist, n_finite, n_total),
                "left":        (min_dist, n_finite, n_total),
                "front_right": (min_dist, n_finite, n_total),
                "right":       (min_dist, n_finite, n_total),
            }
        All distances are in metres; min_dist == inf means no finite
        return in the sector (treat as "open" when the bug is a
        real gap, but never as "blocked").  No fake LEFT proxy is
        used for RIGHT parking — RIGHT parking monitors
        front / front_right / right.
        """
        front = self._bonus_sector_min_dist(
            -self.bonus_park_front_fov_deg,
            +self.bonus_park_front_fov_deg)
        front_left = self._bonus_sector_min_dist(
            self.bonus_park_frontleft_lo_deg,
            self.bonus_park_frontleft_hi_deg)
        left = self._bonus_sector_min_dist(
            self.bonus_park_left_lo_deg,
            self.bonus_park_left_hi_deg)
        front_right = self._bonus_sector_min_dist(
            self.bonus_park_frontright_lo_deg,
            self.bonus_park_frontright_hi_deg)
        right = self._bonus_sector_min_dist(
            self.bonus_park_right_lo_deg,
            self.bonus_park_right_hi_deg)
        return {
            "front": front,
            "front_left": front_left,
            "left": left,
            "front_right": front_right,
            "right": right,
        }

    @staticmethod
    def _bonus_safety_verdict(front_dist, side_dist_1, side_dist_2,
                              caution_m, critical_m, emergency_m,
                              blocked_m):
        """Classify the parking situation from the three relevant
        LiDAR sector minima.

        The caller picks the three sectors to inspect based on the
        latched parking side:
          - LEFT parking  ->  front, front_left, left
          - RIGHT parking ->  front, front_right, right

        Returns one of (in increasing order of urgency):
            "CLEAR"     — every sector is well above the CAUTION
                           threshold.  No safety action needed.
            "CAUTION"   — at least one sector is closer than the
                           CAUTION threshold.  Forward speed is
                           reduced, but the maneuver is NOT aborted.
            "BLOCKED"   — at least one sector is closer than the
                           BLOCKED threshold.  Triggers a full
                           recovery cycle (STOP -> REVERSE ->
                           REALIGN -> RETRY).
            "CRITICAL"  — at least one sector is closer than the
                           CRITICAL threshold.  Forces forward
                           velocity to zero and the turn target
                           to zero.  This is the layer that
                           prevents the "TURN -> HIT" failure.
            "EMERGENCY" — at least one sector is closer than the
                           EMERGENCY threshold.  Forces an
                           immediate STOP (zero velocity, zero
                           turn) even from inside a recovery
                           sub-state.

        The four-tier model is what makes the controller start
        braking and straightening BEFORE the cone is actually
        touched: emergency = "stop now", critical = "stop
        forward, straighten", blocked = "trigger recovery cycle",
        caution = "ease off".
        """
        # Order matters: check the most urgent first.
        if ((math.isfinite(front_dist) and front_dist < emergency_m)
                or (math.isfinite(side_dist_1) and side_dist_1 < emergency_m)
                or (math.isfinite(side_dist_2) and side_dist_2 < emergency_m)):
            return "EMERGENCY"
        if ((math.isfinite(front_dist) and front_dist < critical_m)
                or (math.isfinite(side_dist_1) and side_dist_1 < critical_m)
                or (math.isfinite(side_dist_2) and side_dist_2 < critical_m)):
            return "CRITICAL"
        if ((math.isfinite(front_dist) and front_dist < blocked_m)
                or (math.isfinite(side_dist_1) and side_dist_1 < blocked_m)
                or (math.isfinite(side_dist_2) and side_dist_2 < blocked_m)):
            return "BLOCKED"
        if ((math.isfinite(front_dist) and front_dist < caution_m)
                or (math.isfinite(side_dist_1) and side_dist_1 < caution_m)
                or (math.isfinite(side_dist_2) and side_dist_2 < caution_m)):
            return "CAUTION"
        return "CLEAR"

    # =====================================================================
    # BONUS debug logging (spec format)
    # =====================================================================
    def _bonus_log_status(self, sectors, safety, cmd_speed, cmd_turn,
                          remaining_side, missing_side):
        """Emit the concise 5-10 Hz BONUS status log requested by the spec.

        Format:
            BONUS:
              phase=...
              vectors=...
              remaining_side=...
              missing_side=...
              front=...  front_left=...  front_right=...
              left=...  right=...
              cmd_speed=...  cmd_turn=...
              attempt=...
        """
        front_d, _, _ = sectors["front"]
        fl_d, _, _ = sectors["front_left"]
        fr_d, _, _ = sectors["front_right"]
        left_d, _, _ = sectors["left"]
        right_d, _, _ = sectors["right"]

        def fmt(d):
            if not math.isfinite(d):
                return "inf"
            return f"{d:.2f}"

        # External state string includes the recovery sub-state when
        # one is active, so the operator can see what is happening.
        if self._bonus_recovery_step != BonusRecovery.IDLE:
            state_str = f"{self._bonus_phase} ({self._bonus_recovery_step})"
        else:
            state_str = self._bonus_phase

        # Vector count from the most recent /edge_vectors.
        if self._bonus_last_kind == "TWO":
            vec_count_str = "2"
        elif self._bonus_last_kind == "ONE":
            vec_count_str = f"1 ({remaining_side})"
        elif self._bonus_last_kind == "ZERO":
            vec_count_str = "0"
        else:
            vec_count_str = "?"

        self.get_logger().info(
            "BONUS:\n"
            f"  phase          = {state_str}\n"
            f"  vectors        = {vec_count_str}\n"
            f"  remaining_side = {remaining_side}\n"
            f"  missing_side   = {missing_side}\n"
            f"  front          = {fmt(front_d)} m\n"
            f"  front_left     = {fmt(fl_d)} m\n"
            f"  front_right    = {fmt(fr_d)} m\n"
            f"  left           = {fmt(left_d)} m\n"
            f"  right          = {fmt(right_d)} m\n"
            f"  cmd_speed      = {cmd_speed:+.3f}\n"
            f"  cmd_turn       = {cmd_turn:+.3f}\n"
            f"  attempt        = {self._bonus_parking_attempt_count}/"
            f"{self.bonus_park_max_parking_retries}\n"
            f"  safety         = {safety}"
        )

    def _bonus_log_gap_confirmed(self, missing_side):
        """Emit the BONUS GAP CONFIRMED block once per confirmed gap."""
        self.get_logger().info(
            "================ BONUS GAP CONFIRMED ================\n"
            f"Missing side: {missing_side}\n"
            f"TURNING: {missing_side}\n"
            "======================================================"
        )

    def _bonus_log_vector(self, vector_count, remaining_side,
                          missing_side, action):
        """Emit the BONUS VECTOR block whenever a single vector is
        observed.  This is the verbose per-frame block the spec
        requests for the missing-side case.
        """
        self.get_logger().info(
            "BONUS VECTOR:\n"
            f"  vector_count  = {vector_count}\n"
            f"  remaining_side = {remaining_side}\n"
            f"  missing_side   = {missing_side}\n"
            f"  ACTION         = {action}"
        )

    def _bonus_log_entry(self, side, cmd_speed, cmd_turn):
        """Emit the BONUS ENTRY block at the start of ENTRY."""
        self.get_logger().info(
            "BONUS ENTRY:\n"
            f"  Side   = {side}\n"
            f"  Speed  = {cmd_speed:.2f} m/s\n"
            f"  Turn   = {cmd_turn:+.2f}"
        )

    def _bonus_log_obstacle(self, sectors, action):
        """Emit the BONUS OBSTACLE block whenever a safety action fires."""
        front_d, _, _ = sectors["front"]
        fl_d, _, _ = sectors["front_left"]
        fr_d, _, _ = sectors["front_right"]
        left_d, _, _ = sectors["left"]
        right_d, _, _ = sectors["right"]

        def fmt(d):
            if not math.isfinite(d):
                return "inf"
            return f"{d:.2f}"

        self.get_logger().warn(
            "BONUS OBSTACLE:\n"
            f"  Front       = {fmt(front_d)} m\n"
            f"  Front-Left  = {fmt(fl_d)} m\n"
            f"  Front-Right = {fmt(fr_d)} m\n"
            f"  Left        = {fmt(left_d)} m\n"
            f"  Right       = {fmt(right_d)} m\n"
            f"  Action      = {action}"
        )

    def _bonus_log_recovery_stop(self, sectors):
        """Emit the BONUS RECOVERY: STOP block (one-shot per cycle)."""
        self.get_logger().warn(
            "BONUS RECOVERY:\n"
            "  State = STOP"
        )

    def _bonus_log_recovery_reverse(self, speed, distance,
                                   target_dist, elapsed, target_time):
        """Emit the BONUS RECOVERY: REVERSE block (one-shot per cycle).

        The reverse terminates by TIME (primary) and distance
        (bonus check).  Both progress and targets are shown so
        the operator can see which terminator fired.
        """
        self.get_logger().warn(
            "BONUS REVERSE:\n"
            f"  Command  = {speed:+.3f} m/s\n"
            f"  Duration = {elapsed:.2f} / {target_time:.2f} s\n"
            f"  Distance = {distance:.2f} / {target_dist:.2f} m"
        )

    def _bonus_log_recovery_reverse_complete(self, distance,
                                            target_dist, elapsed,
                                            target_time):
        """Emit the BONUS RECOVERY: REVERSE COMPLETE block."""
        self.get_logger().info(
            "BONUS REVERSE COMPLETE:\n"
            f"  Duration = {elapsed:.2f} / {target_time:.2f} s\n"
            f"  Distance = {distance:.2f} / {target_dist:.2f} m"
        )

    def _bonus_log_recovery_realign(self, side, correction):
        """Emit the BONUS RECOVERY: REALIGN block (one-shot per cycle)."""
        self.get_logger().info(
            "BONUS RECOVERY:\n"
            f"  State      = REALIGN\n"
            f"  Side       = {side}\n"
            f"  Correction = {correction:+.3f}"
        )

    def _bonus_log_retry(self, side, attempt, max_attempts):
        """Emit the BONUS RETRY block (one-shot per retry)."""
        self.get_logger().info(
            "BONUS RETRY:\n"
            f"  Side    = {side}\n"
            f"  Attempt = {attempt}/{max_attempts}"
        )

    def _bonus_log_recovery_retry(self, side, retry_distance):
        """Emit the BONUS RECOVERY: RETRY block (one-shot per cycle)."""
        self.get_logger().info(
            "BONUS RECOVERY:\n"
            f"  State        = RETRY\n"
            f"  Side         = {side}\n"
            f"  Retry target = {retry_distance:.2f} m (straight forward)"
        )

    def _bonus_log_parked(self, side):
        """Emit the BONUS PARKED block (one-shot, on success)."""
        self.get_logger().info(
            "========================================\n"
            "BONUS PARKED SUCCESSFULLY\n"
            f"SIDE: {side}\n"
            "========================================"
        )

    def _bonus_log_failed(self, reason):
        """Emit the BONUS PARKED FAILURE block (one-shot, on failure)."""
        self.get_logger().warn(
            "========================================\n"
            "BONUS PARKING FAILED\n"
            f"Reason: {reason}\n"
            f"Attempts used: {self._bonus_parking_attempt_count}\n"
            "========================================"
        )

    def _bonus_log_parked_success(self, side):
        """Emit the spec-format BONUS PARKING SUCCESS banner.

        The spec explicitly says the log MUST say
        "BONUS PARKING SUCCESS — FULLY INSIDE SLOT" so the
        operator can confirm that the robot physically drove
        deep enough into the slot to be considered parked.
        """
        self.get_logger().info(
            "========================================\n"
            "BONUS PARKING SUCCESS — FULLY INSIDE SLOT\n"
            f"SIDE={side}\n"
            f"PARK_FORWARD distance: "
            f"{self.bonus_park_forward_distance_m:.2f} m\n"
            f"Attempts used: {self._bonus_parking_attempt_count}\n"
            "========================================"
        )

    def _bonus_log_driving_deeper(self, side, sectors, safety,
                                  cmd_speed, driven_m, required_m):
        """Emit the "DRIVING DEEPER INTO SLOT" log block.

        Fires every ~0.2 s while the robot is in
        BONUS_PARK_FORWARD so the operator can watch the
        robot actually drive deep into the parking slot.
        This is the per-spec requirement that the
        PARK_FORWARD log line is continuously visible and
        not a one-shot.
        """
        front_d, _, _ = sectors["front"]
        fl_d, _, _ = sectors["front_left"]
        left_d, _, _ = sectors["left"]

        def fmt(d):
            if not math.isfinite(d):
                return "inf"
            return f"{d:.2f}"

        self.get_logger().info(
            "BONUS PARK_FORWARD:\n"
            f"  Phase        = PARK_FORWARD\n"
            f"  Side         = {side}\n"
            f"  DRIVING DEEPER INTO SLOT\n"
            f"  Park-forward distance = {driven_m:.2f} / "
            f"{required_m:.2f} m\n"
            f"  Speed        = {cmd_speed:+.3f} m/s\n"
            f"  Turn         = 0.000\n"
            f"  Front        = {fmt(front_d)} m\n"
            f"  Front-left   = {fmt(fl_d)} m\n"
            f"  Left         = {fmt(left_d)} m\n"
            f"  Safety       = {safety}\n"
            f"  Attempt      = {self._bonus_parking_attempt_count}/"
            f"{self.bonus_park_max_attempts}"
        )

    def _bonus_log_parking_progress(self, sectors, safety, cmd_speed,
                                    cmd_turn, remaining_side, missing_side):
        """Emit the user-spec parking progress log while in
        PARKING_LEFT / PARKING_RIGHT.

        Format (per the spec):

            BONUS PARKING:
              side=LEFT
              progress=0.18/0.80 m
              speed=+0.05
              turn=+0.70
              front=1.40 m
              front_left=0.60 m
              left=0.52 m
              front_right=inf
              right=inf
              safety=CLEAR
              attempt=1/4

        The progress line is the KEY new field: it is the
        `current_travel_distance - parking_entry_start_distance`
        accumulator.  The robot does NOT declare success until
        progress >= bonus_park_enter_distance_m (one full lane
        width), so the operator can see at a glance how much
        further the robot still has to travel.
        """
        front_d, _, _ = sectors["front"]
        fl_d, _, _ = sectors["front_left"]
        fr_d, _, _ = sectors["front_right"]
        left_d, _, _ = sectors["left"]
        right_d, _, _ = sectors["right"]

        def fmt(d):
            if not math.isfinite(d):
                return "inf"
            return f"{d:.2f}"

        # Parking progress = distance travelled since the parking
        # side was latched.  _bonus_entry_start_travel is set to
        # _travel_dist at the exact moment PARKING_LEFT/RIGHT
        # begins, and again at the end of every recovery cycle.
        # It is NEVER updated by /edge_vectors (no re-decision
        # of the side after parking has started).
        progress = self._travel_dist - self._bonus_entry_start_travel
        required = self.bonus_park_enter_distance_m

        latched = self._bonus_latched_side or "LEFT"

        # External state string includes the recovery sub-state
        # when one is active, so the operator can see what is
        # happening.
        if self._bonus_recovery_step != BonusRecovery.IDLE:
            state_str = f"{self._bonus_phase} ({self._bonus_recovery_step})"
        else:
            state_str = self._bonus_phase

        self.get_logger().info(
            "BONUS PARKING:\n"
            f"  state         = {state_str}\n"
            f"  side          = {latched}\n"
            f"  progress      = {progress:.2f}/{required:.2f} m\n"
            f"  speed         = {cmd_speed:+.3f}\n"
            f"  turn          = {cmd_turn:+.3f}\n"
            f"  front         = {fmt(front_d)} m\n"
            f"  front_left    = {fmt(fl_d)} m\n"
            f"  left          = {fmt(left_d)} m\n"
            f"  front_right   = {fmt(fr_d)} m\n"
            f"  right         = {fmt(right_d)} m\n"
            f"  safety        = {safety}\n"
            f"  attempt       = {self._bonus_parking_attempt_count}/"
            f"{self.bonus_park_max_attempts}\n"
            f"  remaining_side= {remaining_side}\n"
            f"  missing_side  = {missing_side}"
        )

    def _bonus_log_emergency_stop(self):
        """Emit the BONUS OBSTACLE EMERGENCY block (one-shot per cycle)."""
        self.get_logger().warn(
            "BONUS OBSTACLE:\n"
            "  EMERGENCY -> immediate STOP"
        )



    def _bonus_lidar_open_pct(self, side):
        """Return (open_fraction, n_beams_considered) for one side.

        A beam is considered "open" when it is either infinite
        (no return — the laser went past the slot without hitting
        anything) or its finite range is greater than
        bonus_side_clear_m.  NaN and out-of-range finite values
        count as NOT open (they could be a wall right next to the
        robot).  This is the percentage the spec asks for in the
        "LEFT LIDAR OPEN %" / "RIGHT LIDAR OPEN %" debug fields.
        """
        n = self._last_range_count
        if n == 0 or self._last_angle_increment == 0.0:
            return 0.0, 0
        if side == "RIGHT":
            lo = self.bonus_side_offset_deg
            hi = self.bonus_side_offset_deg + self.bonus_clear_fov_deg
        else:
            lo = -(self.bonus_side_offset_deg + self.bonus_clear_fov_deg)
            hi = -self.bonus_side_offset_deg
        lo_rad = math.radians(lo)
        hi_rad = math.radians(hi)
        i_lo = int(round((lo_rad - self._last_angle_min) / self._last_angle_increment))
        i_hi = int(round((hi_rad - self._last_angle_min) / self._last_angle_increment))
        i_lo = max(0, min(n - 1, i_lo))
        i_hi = max(0, min(n - 1, i_hi))
        if i_hi < i_lo:
            i_lo, i_hi = i_hi, i_lo

        total = 0
        open_count = 0
        for i in range(i_lo, i_hi + 1):
            r = self._last_ranges[i % n]
            total += 1
            if math.isinf(r):
                # No return — the laser went past the slot.  Treat as
                # open (the spec wants a "physical gap" here).
                open_count += 1
            elif not math.isfinite(r):
                # NaN — ignore, do NOT count as open.
                pass
            elif r > self.bonus_side_clear_m:
                # Finite range beyond the clearance threshold — open.
                open_count += 1
            else:
                # Finite range <= clearance threshold — blocked.
                pass
        if total <= 0:
            return 0.0, 0
        return open_count / total, total

    def _bonus_slew_steer(self, target):
        """Limit how fast the steering command can change per tick.

        Never aggressive: at 30 Hz the default bonus_turn_slew of 0.04
        means it takes at least 0.04 / 0.04 = 1 s of consistent demand
        to ramp from 0 to a turn of magnitude 1.  Combined with
        bonus_turn_max, the command is always well-behaved.
        """
        prev = self._bonus_turn_cmd
        delta = target - prev
        if delta > self.bonus_turn_slew:
            delta = self.bonus_turn_slew
        elif delta < -self.bonus_turn_slew:
            delta = -self.bonus_turn_slew
        out = prev + delta
        if out > self.bonus_turn_max:
            out = self.bonus_turn_max
        elif out < -self.bonus_turn_max:
            out = -self.bonus_turn_max
        return out

    def _bonus_compute_retry_scales(self):
        """Return (turn_scale, speed_scale) for the current attempt.

        LEGACY: retained for backward-compat with any external
        caller.  The current bonus controller does NOT use scales
        (every retry uses the full base strength).  Returns (1.0, 1.0)
        unconditionally so this function is safe to call but does
        not affect behaviour.
        """
        return 1.0, 1.0

    # =====================================================================
    # BONUS_PARKING_TEST — sign-convention helpers
    # =====================================================================
    # Verified by reading the existing normal line follower's command path:
    #   publish_drive_cmd(final_speed, self.steer_sign * final_turn)
    #   self.steer_sign = -1.0  (default)
    # When the buggy is too far LEFT, the PID outputs final_turn > 0,
    # which becomes axes[3] = steer_sign * final_turn = -1.0 * positive
    # = negative.  Because the buggy turned RIGHT (away from the
    # left-side error), the convention is:
    #
    #   axes[3] > 0  -> physical LEFT
    #   axes[3] < 0  -> physical RIGHT
    #
    # The bonus controller publishes drive commands via
    # `publish_drive_cmd(speed, _bonus_turn_cmd)` directly — without
    # multiplying by `self.steer_sign` (the normal line follower's
    # `steer_sign` multiplication would invert the sign twice and
    # produce the wrong direction).  So we MUST store
    # `_bonus_turn_cmd` directly in the joystick convention:
    #
    #   _bonus_command_left()  -> sets _bonus_turn_cmd to a positive value
    #                            (axes[3] > 0, vehicle turns LEFT)
    #   _bonus_command_right() -> sets _bonus_turn_cmd to a negative value
    #                            (axes[3] < 0, vehicle turns RIGHT)
    #
    # The two helpers below document the sign convention in one place
    # and produce an explicit log line so the operator can verify that
    # the requested side matches the published `axes[3]` value.
    # =====================================================================

    def _bonus_command_left(self, strength):
        """Set `_bonus_turn_cmd` to a positive value so axes[3] > 0
        and the vehicle physically turns LEFT.

        The value is clamped to `bonus_turn_max`.  Slew-limiting is
        applied separately by the caller (via `_bonus_slew_steer`).
        """
        target = +abs(float(strength))
        if target > self.bonus_turn_max:
            target = self.bonus_turn_max
        self._bonus_turn_cmd = target
        self.get_logger().info(
            f"BONUS STEERING: requested=LEFT  published_turn="
            f"{self._bonus_turn_cmd:+.3f}  physical_convention=LEFT "
            f"(axes[3] > 0)"
        )

    def _bonus_command_right(self, strength):
        """Set `_bonus_turn_cmd` to a negative value so axes[3] < 0
        and the vehicle physically turns RIGHT.

        The value is clamped to `bonus_turn_max`.  Slew-limiting is
        applied separately by the caller (via `_bonus_slew_steer`).
        """
        target = -abs(float(strength))
        if target < -self.bonus_turn_max:
            target = -self.bonus_turn_max
        self._bonus_turn_cmd = target
        self.get_logger().info(
            f"BONUS STEERING: requested=RIGHT  published_turn="
            f"{self._bonus_turn_cmd:+.3f}  physical_convention=RIGHT "
            f"(axes[3] < 0)"
        )

    def _bonus_simple_safety(self, front_d, side1_d, side2_d):
        """Per-spec safety verdict for PARKING_LEFT / PARKING_RIGHT.

        The user explicitly required that the side-sector close
        readings (cones on the latched side) MUST NOT cause a STOP
        or recovery cycle.  Only the FRONT sector triggers STOP /
        BLOCKED / CRITICAL / EMERGENCY.  The side sectors only
        contribute to CAUTION (which slows the robot down but does
        not abort parking).

        Returns one of:
            "EMERGENCY" — front < bonus_park_front_emergency_m
                          (immediate STOP, enter recovery)
            "BLOCKED"   — front < bonus_park_front_blocked_m
                          (enter recovery: STOP -> REVERSE -> REALIGN)
            "CRITICAL"  — front < bonus_park_success_clearance_m
                          (force zero forward speed, start recovery)
            "CAUTION"   — any of front / side1 / side2 < some
                          threshold (slow down, keep turning)
            "CLEAR"     — nothing in the way
        """
        # EMERGENCY: front is critically close.  Immediate STOP,
        # enter recovery sub-state.
        if (math.isfinite(front_d)
                and front_d < self.bonus_park_front_emergency_m):
            return "EMERGENCY"
        # BLOCKED: front is in the recovery zone.  Enter recovery.
        if (math.isfinite(front_d)
                and front_d < self.bonus_park_front_blocked_m):
            return "BLOCKED"
        # CRITICAL: front is closer than the success-clearance but
        # not yet at the BLOCKED threshold.  Force zero forward
        # speed and start recovery.
        if (math.isfinite(front_d)
                and front_d < self.bonus_park_success_clearance_m):
            return "CRITICAL"
        # CAUTION: front is OK but a side is closer than ~0.6 m.
        # The side sectors are EXPECTED to be tight during
        # parking (the cones are right next to the robot), so
        # this only slows the robot down.  The threshold is set
        # well below the cone-spacing so we don't false-trigger.
        caution_m = self.bonus_park_success_clearance_m
        if ((math.isfinite(front_d) and front_d < caution_m)
                or (math.isfinite(side1_d) and side1_d < caution_m)
                or (math.isfinite(side2_d) and side2_d < caution_m)):
            return "CAUTION"
        return "CLEAR"

    def _run_bonus_tick(self, now):
        """One control-loop tick of the BONUS_PARKING_TEST state machine.

        This is the ONLY producer of /cerebri/in/joy while in
        BONUS_PARKING_TEST (control_loop returns early when the FSM
        is in this state).  The normal line following, safe-zone
        detector, wall-parallel parking, hospital-stop and obstacle
        avoidance are all bypassed.

        Explicit state machine:

            BONUS_SEARCH
                drive forward slowly; read /edge_vectors every tick.
                When a single vector is observed on the SAME side for
                `bonus_missing_side_frames` consecutive ticks,
                LATCH the OPPOSITE side as the parking side and
                transition to BONUS_APPROACH.  The side is NEVER
                re-evaluated after latching.

            BONUS_APPROACH
                drive STRAIGHT forward for
                `bonus_park_approach_distance_m` (default ~0.30 m)
                at `bonus_park_approach_speed` (default ~0.15 m/s)
                with steering = 0.  Transitions to
                BONUS_FULL_TURN when the distance is covered.

            BONUS_FULL_TURN
                drive forward slowly while turning STRONGLY
                toward the latched side for up to
                `bonus_park_full_turn_max_duration_s` (default 4.0
                s), or until the FRONT is critically close
                (EMERGENCY/BLOCKED/CRITICAL).  Steering is a fixed
                strong value (`bonus_park_full_turn_strength`,
                default ~0.80) in the latched-side direction.
                No lane-vector steering.  Transitions to
                BONUS_STRAIGHTEN when the timer fires or
                recovery starts.

            BONUS_STRAIGHTEN
                steering = 0; continue forward at
                `bonus_park_approach_speed` for
                `bonus_park_straighten_duration_s` (default 0.4
                s).  Transitions to BONUS_PARK_FORWARD.

            BONUS_PARK_FORWARD
                drive STRAIGHT forward (steering = 0) for
                `bonus_park_forward_distance_m` (default 0.75 m)
                at `bonus_park_forward_speed` (default ~0.15
                m/s).  When the distance is covered AND the
                FRONT is reasonably clear (CLEAR/CAUTION), the
                FSM transitions to BONUS_PARKED.

            BONUS_PARKED  (terminal)
                linear.x = 0, steering = 0.  Latched side and
                final progress are logged via
                `_bonus_log_parked_success`.

            BONUS_FAILED  (terminal)
                linear.x = 0, steering = 0.  Reason is logged
                via `_bonus_log_failed`.

        Recovery sub-state machine (entered from any active
        parking phase on EMERGENCY/BLOCKED/CRITICAL):

            BONUS_RECOVERY_STOP
                linear.x = 0 for
                `bonus_park_recovery_stop_duration_s` (default
                0.2 s).  Lets the LiDAR scan catch up.

            BONUS_RECOVERY_REVERSE
                linear.x = `bonus_park_recovery_reverse_speed`
                (negative, default -0.08 m/s) for
                `bonus_park_recovery_reverse_duration_s`
                (default 1.2 s).  Time-based termination
                (primary); distance used as a bonus check.

            BONUS_RECOVERY_STRAIGHTEN
                stop, re-read LiDAR, briefly apply a small
                corrective steering nudge for
                `bonus_park_recovery_straighten_duration_s`
                (default 0.3 s).

            BONUS_RECOVERY_RETRY
                drive STRAIGHT forward (steering = 0) for
                `bonus_park_retry_approach_distance_base_m`
                + (attempt-2) *
                `bonus_park_retry_approach_distance_increment_m`
                (default 0.40 + (attempt-2)*0.10), then
                transition back to BONUS_FULL_TURN on the
                SAME latched side.  Each retry physically
                changes the starting pose because the per-retry
                approach distance is INCREASED.  The retry
                ALWAYS targets the SAME latched side (no
                re-decision of LEFT / RIGHT after a failed
                attempt).

        Sign convention (verified by reading the normal line
        follower's command path):
            * `_bonus_turn_cmd` is in the JOYSTICK convention
              used by the vehicle:
                  positive = physical LEFT
                  negative = physical RIGHT
            * The bonus controller publishes via
              `publish_drive_cmd(speed, _bonus_turn_cmd)`
              directly, WITHOUT multiplying by `self.steer_sign`.
              The normal line follower's `steer_sign`
              multiplication would invert the sign again,
              producing the wrong direction.
            * The two helpers `_bonus_command_left()` and
              `_bonus_command_right()` are the SINGLE source of
              truth for the sign convention.  All turn targets
              flow through them.
        """
        # ============================================================
        # Helper closures for the recovery sub-state machine
        # ============================================================
        #
        # The new recovery sequence (per spec rev 2) is:
        #   STOP -> REVERSE (with COUNTER-STEER)
        #        -> STOP2 -> STRAIGHTEN (short forward straighten)
        #        -> STOP3 -> FORWARD (0.65 m to a NEW starting pose)
        #        -> back to FULL_TURN (same latched side).
        #
        # The REVERSE step uses COUNTER-STEERING (LEFT parking
        # reverses with a RIGHT turn, RIGHT parking reverses with
        # a LEFT turn) so the buggy physically rotates toward
        # a better starting orientation for the next parking
        # attack.  This replaces the old "reverse straight with
        # steering=0" behaviour which only moved the buggy
        # backward without changing its heading.

        def _enter_recovery_stop():
            """Transition into BONUS_RECOVERY_STOP (initial pause)."""
            self._bonus_recovery_step = BonusRecovery.STOP
            self._bonus_recovery_start_time = now
            self._bonus_recovery_start_travel = self._travel_dist
            self._bonus_recovery_stop_logged = False
            self._bonus_recovery_reverse_distance = 0.0
            self._bonus_recovery_last_travel_time = now

        def _enter_recovery_reverse():
            """Transition from STOP to REVERSE (with counter-steer)."""
            self._bonus_recovery_step = BonusRecovery.REVERSE
            self._bonus_recovery_start_time = now
            self._bonus_recovery_start_travel = self._travel_dist
            self._bonus_recovery_reverse_distance = 0.0
            self._bonus_recovery_last_travel_time = now
            self._bonus_recovery_reverse_logged = False
            # Reset the parking phase anchor so a failed
            # PARK_FORWARD distance is NOT carried over after
            # the recovery.  The retry must independently
            # cover the full PARK_FORWARD target.
            self._bonus_phase_start_time = now
            self._bonus_phase_start_travel = self._travel_dist
            self._bonus_park_forward_logged = False

        def _enter_recovery_stop2():
            """Transition from REVERSE to STOP2 (post-reverse pause)."""
            self._bonus_recovery_step = BonusRecovery.STOP2
            self._bonus_recovery_start_time = now
            self._bonus_recovery_start_travel = self._travel_dist
            self._bonus_recovery_reverse_logged = False

        def _enter_recovery_straighten():
            """Transition from STOP2 to STRAIGHTEN
            (short forward alignment nudge)."""
            self._bonus_recovery_step = BonusRecovery.STRAIGHTEN
            self._bonus_recovery_start_time = now
            self._bonus_recovery_start_travel = self._travel_dist
            self._bonus_recovery_straighten_logged = False

        def _enter_recovery_stop3():
            """Transition from STRAIGHTEN to STOP3
            (post-straighten pause)."""
            self._bonus_recovery_step = BonusRecovery.STOP3
            self._bonus_recovery_start_time = now
            self._bonus_recovery_start_travel = self._travel_dist
            self._bonus_recovery_straighten_logged = False

        def _enter_recovery_forward():
            """Transition from STOP3 to FORWARD (drive to a
            NEW starting pose for the retry)."""
            self._bonus_recovery_step = BonusRecovery.FORWARD
            self._bonus_recovery_start_time = now
            self._bonus_recovery_start_travel = self._travel_dist
            self._bonus_recovery_retry_logged = False

        def _enter_full_turn_for_retry():
            """End of recovery: reset phase anchor and transition
            back to FULL_TURN with the SAME latched side.

            The parking attempt counter is incremented HERE
            (not at the start of recovery) so the attempt number
            correctly counts the parking attempts that actually
            executed a FULL_TURN.  After incrementing, if we
            exceed the cap, transition to FAILED."""
            # Bump the attempt counter — this counts the
            # attempt that is about to execute, NOT the failed
            # attempt that triggered recovery.
            self._bonus_parking_attempt_count += 1
            if (self._bonus_parking_attempt_count
                    > self.bonus_park_max_attempts):
                self._bonus_phase = BonusPhase.FAILED
                self._bonus_recovery_step = BonusRecovery.IDLE
                self._bonus_log_failed(
                    f"max attempts ({self.bonus_park_max_attempts}) "
                    "exhausted")
                return False
            # Apply the per-attempt turn scale table per spec
            # (attempts 1-2: 1.00, 3-4: 0.95, 5-6: 0.90).
            # The base turn strength is
            # `bonus_park_full_turn_strength` (default 1.00);
            # we multiply by the scale.  This is the ONLY place
            # the per-attempt turn strength is reduced — the
            # original BASE of 1.00 is preserved across retries
            # and we only ease off by 5-10% on later attempts.
            self._bonus_phase = BonusPhase.FULL_TURN
            self._bonus_recovery_step = BonusRecovery.IDLE
            self._bonus_recovery_stop_logged = False
            self._bonus_recovery_reverse_logged = False
            self._bonus_recovery_straighten_logged = False
            self._bonus_recovery_retry_logged = False
            self._bonus_park_forward_logged = False
            # Reset FULL_TURN phase anchors so the new
            # FULL_TURN starts from a clean baseline.
            self._bonus_phase_start_time = now
            self._bonus_phase_start_travel = self._travel_dist
            self._bonus_full_turn_logged = False
            self._bonus_turn_cmd = 0.0
            # Log RETRY FULL TURN.
            self._bonus_log_retry(
                self._bonus_latched_side or "LEFT",
                self._bonus_parking_attempt_count,
                self.bonus_park_max_attempts)
            return True

        # ============================================================
        # 0) one-shot init
        # ============================================================
        if self._bonus_state_start_time == 0.0:
            self._bonus_state_start_time = now
        if self._bonus_phase_start_time == 0.0:
            self._bonus_phase_start_time = now
            self._bonus_phase_start_travel = self._travel_dist
        if self._bonus_recovery_last_travel_time == 0.0:
            self._bonus_recovery_last_travel_time = now
        if self._bonus_recovery_start_time == 0.0:
            self._bonus_recovery_start_time = now
        if self._bonus_entry_start_travel == 0.0:
            # Will be reset to the actual latch point when
            # the side latches in SEARCH.  We only set it
            # here so the per-tick progress calculation
            # is defined.
            self._bonus_entry_start_travel = self._travel_dist

        # ============================================================
        # 1) overall watchdog (only when not in a recovery sub-state)
        # ============================================================
        elapsed = now - self._bonus_state_start_time
        timeout = self.bonus_parking_timeout_s
        if (elapsed > timeout
                and self._bonus_recovery_step == BonusRecovery.IDLE
                and self._bonus_phase not in (BonusPhase.PARKED,
                                              BonusPhase.FAILED)):
            self._bonus_phase = BonusPhase.FAILED
            self._bonus_recovery_step = BonusRecovery.IDLE
            self._bonus_log_failed(
                "overall watchdog "
                f"({elapsed:.1f}s > {timeout:.1f}s)")
            self.publish_drive_cmd(0.0, 0.0)
            return

        # ============================================================
        # 2) read latest perception
        # ============================================================
        kind, side = self._classify_edge_vector()
        self._bonus_last_kind = kind
        if kind == "ONE":
            self._bonus_last_visible_side = side
        elif kind == "TWO":
            self._bonus_last_visible_side = "BOTH"
        else:
            self._bonus_last_visible_side = "NONE"

        remaining_side = "NONE"
        missing_side = "NONE"
        if kind == "TWO":
            remaining_side = "BOTH"
            missing_side = "NONE"
        elif kind == "ONE" and side in ("LEFT", "RIGHT"):
            remaining_side = side
            missing_side = "RIGHT" if side == "LEFT" else "LEFT"

        # ============================================================
        # 3) Compute LiDAR sectors ONCE per tick and derive safety
        # ============================================================
        sectors = self._bonus_parking_sectors()
        front_d, _, _ = sectors["front"]
        front_left_d, _, _ = sectors["front_left"]
        left_d, _, _ = sectors["left"]
        front_right_d, _, _ = sectors["front_right"]
        right_d, _, _ = sectors["right"]
        latched = self._bonus_latched_side or "LEFT"
        if latched == "LEFT":
            side1_d = front_left_d
            side2_d = left_d
        else:
            side1_d = front_right_d
            side2_d = right_d
        safety = self._bonus_simple_safety(front_d, side1_d, side2_d)
        self._bonus_last_safety = safety

        # ============================================================
        # 4) BONUS_PARKED (terminal)
        # ============================================================
        if self._bonus_phase == BonusPhase.PARKED:
            self.publish_drive_cmd(0.0, 0.0)
            # Emit the BONUS PARKING SUCCESS banner once on entry.
            if not self._bonus_parked_logged:
                self._bonus_log_parked_success(
                    self._bonus_latched_side or "LEFT")
                self._bonus_parked_logged = True
            return

        # ============================================================
        # 5) BONUS_FAILED (terminal)
        # ============================================================
        if self._bonus_phase == BonusPhase.FAILED:
            self.publish_drive_cmd(0.0, 0.0)
            return

        # ============================================================
        # 6) Recovery sub-state handlers (per spec rev 2)
        #
        # Sequence on FRONT EMERGENCY/BLOCKED/CRITICAL:
        #   STOP  (0.25 s)         — pause at zero velocity
        #   REVERSE (0.50 m)       — drive BACKWARD with
        #                             COUNTER-STEER (LEFT parking
        #                             counter-steer RIGHT, RIGHT
        #                             parking counter-steer LEFT)
        #                             to physically rotate the
        #                             buggy toward a better
        #                             starting orientation
        #   STOP2 (0.20 s)         — post-reverse pause
        #   STRAIGHTEN (0.30 m)    — short forward alignment
        #                             nudge to straighten the
        #                             body with respect to the
        #                             slot
        #   STOP3 (0.10 s)         — post-straighten pause
        #   FORWARD (0.65 m)       — drive to a NEW starting
        #                             pose for the retry
        #   -> back to FULL_TURN (same latched side, attempt++)
        #
        # Important: SIDE CONES never trigger recovery — only
        # the FRONT sector.  The recovery is also the only
        # place where the PARK_FORWARD distance is reset, so
        # a failed 0.7 m cannot be combined with a successful
        # 0.6 m on the next attempt.
        # ============================================================
        if self._bonus_recovery_step == BonusRecovery.STOP:
            # Initial pause.  Time-based termination.
            self.publish_drive_cmd(0.0, 0.0)
            if not self._bonus_recovery_stop_logged:
                self._bonus_log_recovery_stop(sectors)
                self._bonus_recovery_stop_logged = True
            if (now - self._bonus_recovery_start_time
                    >= self.bonus_park_recovery_stop_duration_s):
                _enter_recovery_reverse()
            return

        if self._bonus_recovery_step == BonusRecovery.REVERSE:
            # Drive backwards with COUNTER-STEER for ~0.50 m.
            # The reverse is the key change vs the old code:
            # previously the buggy reversed straight (steering=0)
            # which only moved it backward.  Now it reverses WITH
            # a fixed counter-steer so its heading actually
            # changes, giving the next FULL_TURN a better angle
            # of attack on the slot.
            #
            # Termination is by DISTANCE
            # (bonus_park_recovery_reverse_distance_m) OR
            # by hard TIME cap
            # (bonus_park_recovery_reverse_max_duration_s) —
            # whichever comes first.  This prevents both
            # (a) early termination on a tiny movement, and
            # (b) the buggy reversing forever if odometry is
            # unavailable.
            if self._bonus_recovery_last_travel_time > 0.0:
                dt_rev = (now - self._bonus_recovery_last_travel_time)
                if dt_rev <= 0.0 or dt_rev > 0.5:
                    dt_rev = 0.033
                if (self._odom_linear_x is not None
                        and self._odom_time is not None
                        and time.time() - self._odom_time < 0.25):
                    actual_rev_speed = abs(
                        min(0.0, self._odom_linear_x))
                else:
                    actual_rev_speed = abs(
                        self.bonus_park_recovery_reverse_speed)
                self._bonus_recovery_reverse_distance += (
                    actual_rev_speed * dt_rev)
            self._bonus_recovery_last_travel_time = now

            rev_elapsed = now - self._bonus_recovery_start_time
            rev_dist = self._bonus_recovery_reverse_distance
            # Determine reverse target steering per latched side.
            if latched == "LEFT":
                rev_steer = self.bonus_park_recovery_reverse_steering_left
            else:
                rev_steer = self.bonus_park_recovery_reverse_steering_right
            # Clamp to [-1, +1] just in case.
            if rev_steer > 1.0:
                rev_steer = 1.0
            elif rev_steer < -1.0:
                rev_steer = -1.0

            distance_target = (
                self.bonus_park_recovery_reverse_distance_m)
            time_cap = self.bonus_park_recovery_reverse_max_duration_s

            # Termination: either we've covered enough
            # distance OR we've hit the hard time cap.
            if (rev_dist >= distance_target
                    or rev_elapsed >= time_cap):
                if not self._bonus_recovery_reverse_logged:
                    self._bonus_log_recovery_reverse_complete(
                        rev_dist,
                        distance_target,
                        rev_elapsed,
                        rev_dist)
                    self._bonus_recovery_reverse_logged = True
                _enter_recovery_stop2()
                self.publish_drive_cmd(0.0, 0.0)
                return
            # Continue reversing with counter-steer.  This
            # publishes a REAL negative linear.x plus a
            # non-zero turn command (NOT steering=0).
            self.publish_drive_cmd(
                self.bonus_park_recovery_reverse_speed, rev_steer)
            if not self._bonus_recovery_reverse_logged:
                self._bonus_log_recovery_reverse(
                    self.bonus_park_recovery_reverse_speed,
                    rev_dist,
                    distance_target,
                    rev_elapsed,
                    rev_steer)
                self._bonus_recovery_reverse_logged = True
            return

        if self._bonus_recovery_step == BonusRecovery.STOP2:
            # Post-reverse pause.  Time-based termination.
            self.publish_drive_cmd(0.0, 0.0)
            if (now - self._bonus_recovery_start_time
                    >= self.bonus_park_recovery_stop2_duration_s):
                _enter_recovery_straighten()
            return

        if self._bonus_recovery_step == BonusRecovery.STRAIGHTEN:
            # Short forward alignment run (0.30 m) to
            # straighten the body relative to the slot.
            # Speed is 0.10 m/s, steering = 0.  Distance-based
            # termination.
            straighten_distance = (
                self._bonus_recovery_reverse_distance)  # placeholder
            straighten_distance = (
                self.bonus_park_recovery_straighten_distance_m)
            driven_str = (self._travel_dist
                          - self._bonus_recovery_start_travel)
            self.publish_drive_cmd(0.10, 0.0)
            if not self._bonus_recovery_straighten_logged:
                self._bonus_log_recovery_realign(latched, 0.0)
                self._bonus_recovery_straighten_logged = True
            if driven_str >= straighten_distance:
                _enter_recovery_stop3()
                self.publish_drive_cmd(0.0, 0.0)
            return

        if self._bonus_recovery_step == BonusRecovery.STOP3:
            # Post-straighten pause.  Time-based termination.
            self.publish_drive_cmd(0.0, 0.0)
            if (now - self._bonus_recovery_start_time
                    >= self.bonus_park_recovery_stop3_duration_s):
                _enter_recovery_forward()
            return

        if self._bonus_recovery_step == BonusRecovery.FORWARD:
            # Drive STRAIGHT forward (steering=0) for the
            # recovery forward distance (0.65 m).  This moves
            # the buggy to a NEW starting pose so the retry
            # FULL_TURN attacks the slot from a fresh angle.
            forward_distance = (
                self.bonus_park_recovery_forward_distance_m)
            driven_fwd = (self._travel_dist
                          - self._bonus_recovery_start_travel)
            self.publish_drive_cmd(
                self.bonus_park_recovery_forward_speed, 0.0)
            if not self._bonus_recovery_retry_logged:
                self._bonus_log_recovery_retry(
                    latched, forward_distance)
                self._bonus_recovery_retry_logged = True
            if driven_fwd >= forward_distance:
                # End of forward.  End of recovery.  Transition
                # back to FULL_TURN on the SAME latched side.
                # The attempt counter is bumped here (only
                # at the start of a fresh parking attempt).
                _enter_full_turn_for_retry()
                # _enter_full_turn_for_retry() may have set
                # the phase to FAILED if the cap was hit.
                # In that case, publish zero velocity and
                # return so the rest of the tick doesn't
                # fall through into the FULL_TURN handler.
                if self._bonus_phase == BonusPhase.FAILED:
                    self.publish_drive_cmd(0.0, 0.0)
                    return
                # The turn command will be issued on the
                # next tick by the FULL_TURN handler.  For
                # now, hold zero velocity so the buggy
                # doesn't continue forward with the
                # recovery forward speed.
                self.publish_drive_cmd(0.0, 0.0)
                return
            return

        # ============================================================
        # 7) BONUS_SEARCH — drive forward, latch the missing
        #    side, then transition to BONUS_APPROACH
        # ============================================================
        if self._bonus_phase == BonusPhase.SEARCH:
            # Update the missing-side streak.  Reset on any
            # non-matching frame (TWO / ZERO / CENTER).
            if kind == "ONE" and side in ("LEFT", "RIGHT"):
                if self._bonus_missing_side == missing_side:
                    self._bonus_missing_streak += 1
                else:
                    self._bonus_missing_side = missing_side
                    self._bonus_missing_streak = 1
                if missing_side == "LEFT":
                    self._bonus_left_missing_count += 1
                    self._bonus_right_missing_count = 0
                else:
                    self._bonus_right_missing_count += 1
                    self._bonus_left_missing_count = 0
            else:
                self._bonus_missing_side = None
                self._bonus_missing_streak = 0
                self._bonus_left_missing_count = 0
                self._bonus_right_missing_count = 0

            # Throttled SEARCH status log (~5 Hz).
            if now - self._bonus_log_time >= 0.2:
                self._bonus_log_time = now
                self._bonus_log_status(
                    sectors, safety,
                    cmd_speed=self.bonus_park_search_speed,
                    cmd_turn=0.0,
                    remaining_side=remaining_side,
                    missing_side=missing_side,
                )

            # Streak reached the persistence threshold?
            if (self._bonus_missing_side is not None
                    and self._bonus_missing_streak
                        >= self.bonus_missing_side_frames):
                # LiDAR sanity check: front must have some
                # clearance.  Otherwise the robot would
                # just crash into a wall when it starts
                # turning.
                front_d_local, _, _ = sectors["front"]
                front_safe = (
                    math.isinf(front_d_local)
                    or front_d_local > self.bonus_park_front_blocked_m)
                if front_safe:
                    # LATCH the side and transition to
                    # BONUS_APPROACH.
                    self._bonus_latched_side = self._bonus_missing_side
                    self._bonus_phase = BonusPhase.APPROACH
                    self._bonus_entry_start_travel = self._travel_dist
                    self._bonus_phase_start_time = now
                    self._bonus_phase_start_travel = self._travel_dist
                    # First attempt: full strength, full speed.
                    self._bonus_parking_attempt_count = 1
                    # Reset per-attempt recovery bookkeeping.
                    self._bonus_recovery_step = BonusRecovery.IDLE
                    self._bonus_recovery_start_time = 0.0
                    self._bonus_recovery_stop_logged = False
                    self._bonus_recovery_reverse_logged = False
                    self._bonus_recovery_straighten_logged = False
                    self._bonus_recovery_retry_logged = False
                    self._bonus_parked_logged = False
                    self._bonus_approach_logged = False
                    self._bonus_full_turn_logged = False
                    self._bonus_straighten_logged = False
                    self._bonus_park_forward_logged = False
                    self._bonus_turn_cmd = 0.0
                    self._bonus_log_gap_confirmed(
                        self._bonus_latched_side)
                    self.get_logger().info(
                        f"BONUS: gap confirmed -> "
                        f"BONUS_APPROACH "
                        f"(side={self._bonus_latched_side}, "
                        f"streak={self._bonus_missing_streak})")
                else:
                    # Front blocked: don't latch yet.  Keep
                    # driving forward and waiting for a
                    # cleaner window.
                    self.get_logger().warn(
                        f"BONUS: gap streak ready but FRONT is "
                        f"blocked "
                        f"({sectors['front'][0]:.2f}m <= "
                        f"{self.bonus_park_front_blocked_m:.2f}m); "
                        "holding SEARCH until the front clears.")

            # Still in SEARCH: drive forward slowly with no
            # turn.  Wait for the streak to reach the
            # persistence threshold.
            if self._bonus_phase == BonusPhase.SEARCH:
                self.publish_drive_cmd(
                    self.bonus_park_search_speed, 0.0)
                return

        # ============================================================
        # 8) BONUS_APPROACH — drive STRAIGHT forward for
        #    `bonus_park_approach_distance_m` at
        #    `bonus_park_approach_speed`.
        # ============================================================
        if self._bonus_phase == BonusPhase.APPROACH:
            # FRONT collision safety: enter recovery.  We
            # do NOT re-evaluate the side here; we just
            # suspend the parking control.
            if safety in ("EMERGENCY", "BLOCKED", "CRITICAL"):
                if safety == "EMERGENCY":
                    self._bonus_log_emergency_stop()
                else:
                    self._bonus_log_obstacle(
                        sectors, f"STOP at APPROACH ({safety})")
                self._bonus_turn_cmd = 0.0
                _enter_recovery_stop()
                self.publish_drive_cmd(0.0, 0.0)
                return

            driven_approach = (
                self._travel_dist - self._bonus_phase_start_travel)
            self.publish_drive_cmd(
                self.bonus_park_approach_speed, 0.0)
            if not self._bonus_approach_logged:
                self._bonus_log_entry(
                    latched,
                    self.bonus_park_approach_speed, 0.0)
                self._bonus_approach_logged = True
            # Throttled progress log
            if now - self._bonus_log_time >= 0.2:
                self._bonus_log_time = now
                self._bonus_log_parking_progress(
                    sectors, safety,
                    cmd_speed=self.bonus_park_approach_speed,
                    cmd_turn=0.0,
                    remaining_side=remaining_side,
                    missing_side=missing_side,
                )
            if driven_approach >= self.bonus_park_approach_distance_m:
                # Done with approach.  Transition to FULL_TURN.
                self._bonus_phase = BonusPhase.FULL_TURN
                self._bonus_phase_start_time = now
                self._bonus_phase_start_travel = self._travel_dist
                self._bonus_full_turn_logged = False
            return

        # ============================================================
        # 9) BONUS_FULL_TURN — drive forward while turning
        #    STRONGLY toward the latched side for up to
        #    `bonus_park_full_turn_max_duration_s`.
        #
        # Per spec rev 2:
        #  - LEFT  parking -> steering = +1.00 (full lock, axes[3] > 0)
        #  - RIGHT parking -> steering = -1.00 (full lock, axes[3] < 0)
        #  - The turn command is set IMMEDIATELY to the full
        #    value (no slew-limit, no proportional easing).
        #  - On retries 1-2 the multiplier is 1.00, on 3-4 it
        #    drops to 0.95, on 5-6 to 0.90 — only a small
        #    easing, never a weak turn.
        # ============================================================
        if self._bonus_phase == BonusPhase.FULL_TURN:
            # FRONT collision safety: enter recovery.
            if safety in ("EMERGENCY", "BLOCKED", "CRITICAL"):
                if safety == "EMERGENCY":
                    self._bonus_log_emergency_stop()
                else:
                    self._bonus_log_obstacle(
                        sectors, f"STOP at FULL_TURN ({safety})")
                self._bonus_turn_cmd = 0.0
                _enter_recovery_stop()
                self.publish_drive_cmd(0.0, 0.0)
                return

            # Per-attempt turn strength scale (per spec):
            #   attempts 1-2 -> 1.00
            #   attempts 3-4 -> 0.95
            #   attempts 5-6 -> 0.90
            attempt = self._bonus_parking_attempt_count
            if attempt <= 2:
                turn_scale = 1.00
            elif attempt <= 4:
                turn_scale = 0.95
            else:
                turn_scale = 0.90

            # Set the strong side-directed turn command
            # IMMEDIATELY to the full value.  Per spec, do NOT
            # slew-limit, do NOT ease in.  The base is
            # `bonus_park_full_turn_strength` (default 1.00)
            # multiplied by the per-attempt scale.
            base_strength = self.bonus_park_full_turn_strength
            target_turn = base_strength * turn_scale
            if target_turn > 1.0:
                target_turn = 1.0
            elif target_turn < 0.0:
                target_turn = 0.0
            if latched == "LEFT":
                # +axes[3] = physical LEFT.
                self._bonus_command_left(target_turn)
            else:
                # -axes[3] = physical RIGHT.
                self._bonus_command_right(target_turn)
            # IMPORTANT: do NOT slew-limit the steering in
            # FULL_TURN.  The spec explicitly requires the
            # buggy to reach the full steering command
            # IMMEDIATELY.  _bonus_command_left/right has
            # already set self._bonus_turn_cmd to the full
            # target; we use that value directly below
            # (no _bonus_slew_steer() call here).

            # Apply CAUTION -> halve the forward speed.
            cmd_speed_full_turn = self.bonus_park_full_turn_speed
            if safety == "CAUTION":
                cmd_speed_full_turn *= 0.5

            self.publish_drive_cmd(
                cmd_speed_full_turn, self._bonus_turn_cmd)
            if not self._bonus_full_turn_logged:
                self._bonus_log_entry(
                    latched,
                    cmd_speed_full_turn,
                    self._bonus_turn_cmd)
                self._bonus_full_turn_logged = True
            if now - self._bonus_log_time >= 0.2:
                self._bonus_log_time = now
                self._bonus_log_parking_progress(
                    sectors, safety,
                    cmd_speed=cmd_speed_full_turn,
                    cmd_turn=self._bonus_turn_cmd,
                    remaining_side=remaining_side,
                    missing_side=missing_side,
                )
            # Transition to STRAIGHTEN when the timer fires.
            full_turn_elapsed = now - self._bonus_phase_start_time
            if full_turn_elapsed >= self.bonus_park_full_turn_max_duration_s:
                self._bonus_phase = BonusPhase.STRAIGHTEN
                self._bonus_phase_start_time = now
                self._bonus_phase_start_travel = self._travel_dist
                self._bonus_straighten_logged = False
                self._bonus_turn_cmd = 0.0
            return

        # ============================================================
        # 10) BONUS_STRAIGHTEN — steering = 0, continue forward
        #     for `bonus_park_straighten_duration_s`.
        # ============================================================
        if self._bonus_phase == BonusPhase.STRAIGHTEN:
            # FRONT collision safety: enter recovery.
            if safety in ("EMERGENCY", "BLOCKED", "CRITICAL"):
                if safety == "EMERGENCY":
                    self._bonus_log_emergency_stop()
                else:
                    self._bonus_log_obstacle(
                        sectors, f"STOP at STRAIGHTEN ({safety})")
                self._bonus_turn_cmd = 0.0
                _enter_recovery_stop()
                self.publish_drive_cmd(0.0, 0.0)
                return

            self._bonus_turn_cmd = 0.0
            self.publish_drive_cmd(
                self.bonus_park_approach_speed, 0.0)
            if not self._bonus_straighten_logged:
                self._bonus_log_entry(
                    latched,
                    self.bonus_park_approach_speed, 0.0)
                self._bonus_straighten_logged = True
            if now - self._bonus_log_time >= 0.2:
                self._bonus_log_time = now
                self._bonus_log_parking_progress(
                    sectors, safety,
                    cmd_speed=self.bonus_park_approach_speed,
                    cmd_turn=0.0,
                    remaining_side=remaining_side,
                    missing_side=missing_side,
                )
            straighten_elapsed = now - self._bonus_phase_start_time
            if straighten_elapsed >= self.bonus_park_straighten_duration_s:
                # Done straightening.  Transition to
                # PARK_FORWARD.
                self._bonus_phase = BonusPhase.PARK_FORWARD
                self._bonus_phase_start_time = now
                self._bonus_phase_start_travel = self._travel_dist
                self._bonus_park_forward_logged = False
            return

        # ============================================================
        # 11) BONUS_PARK_FORWARD — drive STRAIGHT forward for
        #     `bonus_park_forward_distance_m` at
        #     `bonus_park_forward_speed`.  Reaching the
        #     distance AND FRONT being clear transitions to
        #     BONUS_PARKED.
        # ============================================================
        if self._bonus_phase == BonusPhase.PARK_FORWARD:
            # FRONT collision safety: enter recovery.  This
            # is the primary safety gate — side cones
            # NEVER cause a STOP or recovery.
            if safety in ("EMERGENCY", "BLOCKED", "CRITICAL"):
                if safety == "EMERGENCY":
                    self._bonus_log_emergency_stop()
                else:
                    self._bonus_log_obstacle(
                        sectors, f"STOP at PARK_FORWARD ({safety})")
                self._bonus_turn_cmd = 0.0
                _enter_recovery_stop()
                self.publish_drive_cmd(0.0, 0.0)
                return

            self._bonus_turn_cmd = 0.0
            # Apply CAUTION -> halve the forward speed.
            cmd_speed_pf = self.bonus_park_forward_speed
            if safety == "CAUTION":
                cmd_speed_pf *= 0.5

            self.publish_drive_cmd(cmd_speed_pf, 0.0)
            if not self._bonus_park_forward_logged:
                self._bonus_log_entry(
                    latched,
                    cmd_speed_pf, 0.0)
                self._bonus_park_forward_logged = True
            # Throttled progress log.  Emit the dedicated
            # "DRIVING DEEPER INTO SLOT" message on every
            # log tick so the operator can watch the bug
            # actually drive forward into the slot.  This
            # is the KEY log per the spec — it must
            # continue to fire throughout the PARK_FORWARD
            # phase and not stop at STRAIGHTEN.
            if now - self._bonus_log_time >= 0.2:
                self._bonus_log_time = now
                self._bonus_log_parking_progress(
                    sectors, safety,
                    cmd_speed=cmd_speed_pf,
                    cmd_turn=0.0,
                    remaining_side=remaining_side,
                    missing_side=missing_side,
                )
                self._bonus_log_driving_deeper(
                    latched, sectors, safety, cmd_speed_pf,
                    self._travel_dist - self._bonus_phase_start_travel,
                    self.bonus_park_forward_distance_m,
                )
            # Success criterion: forward distance covered
            # AND front reasonably clear.  The forward
            # distance is the PRIMARY criterion — the robot
            # MUST have physically driven deep into the
            # slot before this transition fires.  STRAIGHTEN
            # alone is NOT enough.
            driven_pf = (self._travel_dist
                         - self._bonus_phase_start_travel)
            front_clear = (
                math.isfinite(front_d)
                and front_d
                    >= self.bonus_park_success_clearance_m)
            if (driven_pf >= self.bonus_park_forward_distance_m
                    and front_clear
                    and safety in ("CLEAR", "CAUTION")):
                self._bonus_phase = BonusPhase.PARKED
                self._bonus_recovery_step = BonusRecovery.IDLE
                self.publish_drive_cmd(0.0, 0.0)
                if not self._bonus_parked_logged:
                    self._bonus_log_parked_success(
                        self._bonus_latched_side)
                    self._bonus_parked_logged = True
                return
            return

        # ============================================================
        # 12) Defensive safety: an unexpected phase value.
        # ============================================================
        self.get_logger().warn(
            f"BONUS: unexpected phase {self._bonus_phase!r} — stopping")
        self.publish_drive_cmd(0.0, 0.0)

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
