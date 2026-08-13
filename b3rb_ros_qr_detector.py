# Copyright 2024-2026 NXP
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from std_msgs.msg import Bool
import cv2
import numpy as np
from datetime import datetime

# Municipality Server communication message.
from synapse_msgs.msg import ServerCommunication

try:
    from pyzbar import pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    pyzbar = None
    PYZBAR_AVAILABLE = False

# ---------------------------------------------------------------------------
# OFFICIAL NXP PROTOCOL CONSTANTS  (UNCHANGED)
# ---------------------------------------------------------------------------
SERVER_FIELD_SRC = 'src'
SERVER_FIELD_DEST = 'dest'
SERVER_FIELD_UID = 'uid'
SERVER_FIELD_ACK = 'ack'
SERVER_FIELD_MSG = 'msg'

BUGGY_ID = 1
SERVER_ID = 2
UID_MAX = 255
ACK_TIMEOUT_SEC = 2.0
MAX_RETRIES = 5

COMM_IDLE = "IDLE"
COMM_WAITING_ACK = "WAITING_ACK"
COMM_WAITING_SERVER_ASSIGNMENT = "WAITING_SERVER_ASSIGNMENT"
COMM_RETRY = "RETRY"
COMM_FAILED = "FAILED"

MISSION_PATIENT = "PATIENT"
MISSION_HOSPITAL = "HOSPITAL"
MISSION_COMPLETE = "MISSION_COMPLETE"

GOAL_TO_PATIENT_NAME = {
    "A": "PATIENT_1",
    "B": "PATIENT_2",
    "C": "PATIENT_3",
}
PATIENT_NAME_TO_GOAL = {v: k for k, v in GOAL_TO_PATIENT_NAME.items()}

GOAL_TO_HOSPITAL_NAME = {
    "X": "HOSPITAL_1",
    "Y": "HOSPITAL_2",
    "Z": "HOSPITAL_3",
}
HOSPITAL_NAME_TO_GOAL = {v: k for k, v in GOAL_TO_HOSPITAL_NAME.items()}

GOAL_TO_PATIENT_NUM = {"A": 1, "B": 2, "C": 3}
GOAL_TO_HOSPITAL_NUM = {"X": 4, "Y": 5, "Z": 6}

VALID_MISSION_PAYLOADS = {"A", "B", "C", "X", "Y", "Z", "OK", "INVALID"}

# Bonus parking: resend PARKED on this interval until the server says OK.
PARKED_INTERVAL_SEC = 5.0
# 3 patient picks (A/B/C) + 3 hospital drops (X/Y/Z).
TOTAL_MISSION_LEGS = 6

# Frames in a row without the verified QR before we say it left the camera.
QR_GONE_CONFIRM_FRAMES = 6

# ---------------------------------------------------------------------------
# MISSION FINITE STATE MACHINE  (UNCHANGED)
# ---------------------------------------------------------------------------
#
#   SEARCHING_QR  --(correct QR verified once)--> QR_VERIFIED
#   QR_VERIFIED   --(/safe_zone=True + qr_verified)--> SEND_PACKET
#   SEND_PACKET   --(ACK success / next assignment)--> SEARCHING_QR
#
# The FSM starts directly in SEARCHING_QR (initial mission Goal A -> PATIENT_1).
# There is no WAIT_ASSIGNMENT state after startup.
FSM_SEARCHING_QR = "SEARCHING_QR"
FSM_QR_VERIFIED = "QR_VERIFIED"
FSM_SEND_PACKET = "SEND_PACKET"


class QRDetector(Node):
    """
    QR Detector - file 2 topics + file 1 final-mission PARKED protocol.

    File 2 topics (all preserved):
      Sub: /camera/image_raw/compressed
      Sub: /safe_zone
      Sub: /target_type          (loopback, log only)
      Sub: /target_qr            (loopback, log only)
      Sub: /ServerCommunication
      Sub: /mission/turn         (log only)
      Pub: /qr_detection
      Pub: /target_qr
      Pub: /target_type
      Pub: /resume_line_following
      Pub: /mission/available
      Pub: /ServerCommunication

    Extra (does not replace any file-2 topic):
      Pub: /qr_not_visible       (Bool, like /safe_zone — last matching QR left)

    File 1 protocol added:
      Final hospital (goal Z): /safe_zone=True sends "PARKED" once
      instead of the goal letter. Every other mission still sends A/B/C/X/Y.
    """

    def __init__(self):
        super().__init__('qr_detector')

        # Message counters for debug logging
        self.packets_sent = 0
        self.ack_received = 0
        self.ack_failed = 0
        self.retries_total = 0
        self.duplicate_packets = 0
        self.duplicate_acks = 0
        self.current_target_type_for_log = "PATIENT"

        # -----------------------------------------------------------------
        # SINGLE SOURCE OF TRUTH: the current mission.
        # -----------------------------------------------------------------
        self.fsm_state = FSM_SEARCHING_QR

        self.current_goal = "A"
        self.expected_qr = GOAL_TO_PATIENT_NAME[self.current_goal]  # "PATIENT_1"
        self.expected_target_type = "PATIENT"

        self.mission_state = MISSION_PATIENT
        self.current_patient = GOAL_TO_PATIENT_NAME[self.current_goal]
        self.current_patient_id = GOAL_TO_PATIENT_NUM[self.current_goal]
        self.assigned_hospital = None
        self.assigned_hospital_id = None
        self.assigned_hospital_goal = None

        # QR verification (one source of truth: qr_verified).
        self.qr_verified = False
        self.verified_qr = None

        # File 1 leftover flag (Z no longer auto-PARKED; bonus does).
        self.parked_sent = False

        # Bonus: count each completed pick/drop (A,X,B,Y,C,Z) up to 6.
        self.mission_leg_count = 0
        self._completed_legs = set()
        self.bonus_active = False
        self.bonus_ok = False
        self._last_parked_send = None
        self._parked_attempts = 0

        # Visibility of the successfully-verified QR (print + /qr_not_visible).
        self._reset_verified_visibility()

        # -----------------------------------------------------------------
        # Communication state machine (ACK / retry / UID) - unchanged.
        # -----------------------------------------------------------------
        self.comm_state = COMM_IDLE
        self.uid_counter = 0
        self.pending_outgoing_msg = None
        self.pending_send_time = None
        self.retry_count = 0
        self.last_processed_server_uid = None
        self.current_target = None

        self.get_logger().info(
            "\n==========================================\n\n"
            "QR Detector Started\n\n"
            "Communication Node Initialized\n\n"
            "Current Patient : PATIENT_1\n\n"
            "Current Goal    : A\n\n"
            "Waiting for:\n\n"
            "- Municipality Server assignment\n\n"
            "- /safe_zone\n\n"
            "==========================================\n"
        )

        # ------------------- ROS plumbing (file 2 topics, all kept) ------
        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10)
        self.get_logger().info("Subscribed:\n    /camera/image_raw/compressed")

        self.publisher_qr = self.create_publisher(String, '/qr_detection', 10)
        self.get_logger().info("Publisher Ready:\n    /qr_detection")

        self.subscription_safe_zone = self.create_subscription(
            Bool, '/safe_zone', self.safe_zone_callback, 10)
        self.get_logger().info("Subscribed:\n    /safe_zone")

        self.publisher_target_qr = self.create_publisher(String, '/target_qr', 10)
        self.get_logger().info("Publisher Ready:\n    /target_qr")

        self.publisher_target_type = self.create_publisher(String, '/target_type', 10)
        self.get_logger().info("Publisher Ready:\n    /target_type")

        # Loopback observers of qr_detector's own publishes (log only).
        self.subscription_target_type = self.create_subscription(
            String, '/target_type', self.target_type_callback, 10)
        self.get_logger().info("Subscribed:\n    /target_type")

        self.subscription_target_qr = self.create_subscription(
            String, '/target_qr', self.target_qr_callback, 10)
        self.get_logger().info("Subscribed:\n    /target_qr")

        self.publisher_resume = self.create_publisher(String, '/resume_line_following', 10)
        self.get_logger().info("Publisher Ready:\n    /resume_line_following")

        self.publisher_mission_available = self.create_publisher(String, '/mission/available', 10)
        self.get_logger().info("Publisher Ready:\n    /mission/available")

        self.publisher_server = self.create_publisher(ServerCommunication, '/ServerCommunication', 10)
        self.get_logger().info("Publisher Ready:\n    /ServerCommunication")

        self.subscription_server = self.create_subscription(
            ServerCommunication, '/ServerCommunication', self.server_communication_callback, 10)
        self.get_logger().info("Subscribed:\n    /ServerCommunication")

        self.subscription_mission_turn = self.create_subscription(
            String, '/mission/turn', self.mission_turn_callback, 10)
        self.get_logger().info("Subscribed:\n    /mission/turn")

        # Extra publisher (file 2 topics untouched). Line follower parks on this.
        self.publisher_qr_not_visible = self.create_publisher(
            Bool, '/qr_not_visible', 10)
        self.get_logger().info("Publisher Ready:\n    /qr_not_visible")

        self.publisher_bonus = self.create_publisher(String, '/bonus', 10)
        self.get_logger().info("Publisher Ready:\n    /bonus")

        # --- QR vision resources (created once, reused per frame) --------
        self.qr_detector = cv2.QRCodeDetector()
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        if not PYZBAR_AVAILABLE:
            self.get_logger().warn("pyzbar not available. Falling back to cv2.QRCodeDetector only.")

        self.comm_timer = self.create_timer(0.1, self._communication_timeout_check)
        self.bonus_timer = self.create_timer(0.5, self._bonus_parked_tick)

        self.get_logger().info("QR Detector Node started. Waiting for images...")
        self.get_logger().info(
            "====================================\n"
            f"Initial Mission\n"
            f"Patient : {self.current_patient} (Goal {self.current_goal})\n"
            f"Mission State : {self.mission_state}\n"
            f"Comm State    : {self.comm_state}\n"
            f"FSM State     : {self.fsm_state}\n"
            "Protocol: src=1 dest=2 for buggy->server, src=2 dest=1 for server->buggy\n"
            "Final mission (goal Z) sends PARKED on /safe_zone (once).\n"
            "====================================")

        self._log_counters("NODE STARTUP - SEARCHING_QR")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _get_timestamp_str(self):
        try:
            now = self.get_clock().now()
            sec = now.nanoseconds / 1e9
            wall = datetime.now().isoformat()
            return f"{wall} | ROS {sec:.3f}s"
        except Exception:
            return datetime.now().isoformat()

    def _log_ros_topic(self, topic_name, data):
        self.get_logger().info(
            f"\n----------------------------------\n"
            f"ROS TOPIC RECEIVED\n"
            f"Topic: {topic_name}\n"
            f"Timestamp: {self._get_timestamp_str()}\n"
            f"Data: {data}\n"
            f"----------------------------------\n"
        )

    def _is_mission_active(self):
        """The mission is active while we are searching / verified / waiting ACK."""
        return self.fsm_state in (FSM_SEARCHING_QR, FSM_QR_VERIFIED, FSM_SEND_PACKET)

    def _is_final_mission(self):
        """True only for the last hospital delivery in the chain
        (A->X->B->Y->C->Z->OK). Goal "Z" is the final leg before
        MISSION_COMPLETE, so that is the point where /safe_zone=True
        should be converted into the "PARKED" protocol message instead
        of the normal goal-letter packet."""
        return (self.mission_state == MISSION_HOSPITAL
                and self.current_goal == "Z")

    def _note_leg_complete(self, goal):
        """Patient pick or hospital drop. 6 legs → bonus."""
        goal = str(goal or "").strip().upper()
        if goal not in ("A", "B", "C", "X", "Y", "Z"):
            return
        if goal in self._completed_legs:
            return
        self._completed_legs.add(goal)
        self.mission_leg_count = len(self._completed_legs)
        kind = ("patient pick" if goal in ("A", "B", "C")
                else "hospital drop")
        banner = (
            f"*** LEG {self.mission_leg_count}/{TOTAL_MISSION_LEGS}  {kind}  "
            f"goal={goal}\n"
            f"    done={sorted(self._completed_legs)}")
        self.get_logger().info(banner)
        print(banner, flush=True)
        if self.mission_leg_count >= TOTAL_MISSION_LEGS:
            self._start_bonus()

    def _start_bonus(self):
        if self.bonus_active or self.bonus_ok:
            return
        self.bonus_active = True
        self.bonus_ok = False
        self._last_parked_send = None
        self._parked_attempts = 0
        msg = String()
        msg.data = "BONUS"
        self.publisher_bonus.publish(msg)
        banner = (
            "==========================================\n"
            f"BONUS  all {TOTAL_MISSION_LEGS} legs done (3 pick + 3 drop)\n"
            "Published /bonus = BONUS\n"
            "Line follower: 2-lane reverse+straight park\n"
            f"Then PARKED to server every {PARKED_INTERVAL_SEC:.0f} s "
            "until OK\n"
            "==========================================")
        self.get_logger().info(banner)
        print(banner, flush=True)

    def _bonus_parked_tick(self):
        """While bonus is active, send PARKED every 5 s until server OK.

        Self-driving: it does not need /safe_zone to start. Once /bonus has
        fired, PARKED goes out every 5 s and keeps going on INVALID; only a
        server "OK" stops it.
        """
        if not self.bonus_active or self.bonus_ok:
            return
        now = time.time()
        if (self._last_parked_send is not None and
                now - self._last_parked_send < PARKED_INTERVAL_SEC):
            return
        if self.comm_state == COMM_WAITING_ACK:
            return
        self._parked_attempts += 1
        self.get_logger().info(
            f"BONUS: publishing PARKED to /ServerCommunication "
            f"(attempt {self._parked_attempts}, every "
            f"{PARKED_INTERVAL_SEC:.0f} s until OK)")
        self.send_mission_to_server("PARKED")
        self._last_parked_send = now

    def _log_counters(self, context=""):
        self.get_logger().info(
            f"\n==========================================\n"
            f"MESSAGE COUNTERS {context}\n"
            f"Packets Sent: {self.packets_sent}\n"
            f"ACK Received: {self.ack_received}\n"
            f"ACK Failed: {self.ack_failed}\n"
            f"Retries: {self.retries_total}\n"
            f"Duplicate Packets: {self.duplicate_packets}\n"
            f"Duplicate ACKs: {self.duplicate_acks}\n"
            f"Mission Active: {self._is_mission_active()}\n"
            f"FSM: {self.fsm_state} | Comm: {self.comm_state}\n"
            f"Expected QR: {self.expected_qr} | Verified: {self.verified_qr} "
            f"(qr_verified={self.qr_verified})\n"
            f"Legs done: {self.mission_leg_count}/{TOTAL_MISSION_LEGS} "
            f"{sorted(self._completed_legs)}\n"
            f"Bonus: active={self.bonus_active} ok={self.bonus_ok}\n"
            f"==========================================\n"
        )

    @staticmethod
    def _extract_hospital_name(qr_data):
        match = re.search(r'(HOSPITAL_\w+)', qr_data, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return None

    @staticmethod
    def _extract_patient_name(qr_data):
        match = re.search(r'(PATIENT_\w+)', qr_data, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return None

    def _normalize_decoded_qr(self, qr_data):
        if not qr_data:
            return None
        raw = str(qr_data).strip()
        if not raw:
            return None
        detected_patient = self._extract_patient_name(raw)
        detected_hospital = self._extract_hospital_name(raw)
        return detected_patient or detected_hospital or raw.upper()

    def _reset_verified_visibility(self):
        self._verified_visible = False
        self._verified_lost_frames = 0
        self._verified_seen_frames = 0
        self._verified_first_seen_time = None
        self._verified_last_seen_time = None
        self._verified_last_seen_stamp = None
        self._verified_not_visible_logged = False
        if hasattr(self, 'publisher_qr_not_visible'):
            self._publish_qr_not_visible(False)

    def _publish_qr_not_visible(self, gone):
        if not hasattr(self, 'publisher_qr_not_visible'):
            return
        msg = Bool()
        msg.data = bool(gone)
        try:
            self.publisher_qr_not_visible.publish(msg)
            self.get_logger().info(
                f"Publishing /qr_not_visible: {bool(gone)}")
        except Exception as e:
            self.get_logger().error(f"/qr_not_visible publish failed: {e}")

    def _note_verified_qr_seen(self):
        now = time.time()
        if self._verified_first_seen_time is None:
            self._verified_first_seen_time = now
        self._verified_visible = True
        self._verified_lost_frames = 0
        self._verified_seen_frames += 1
        self._verified_last_seen_time = now
        self._verified_last_seen_stamp = self._get_timestamp_str()

    def _update_verified_qr_visibility(self, qr_data):
        """After a successful match, publish once when that QR leaves camera."""
        if not self.qr_verified or not self.verified_qr:
            return

        detected = self._normalize_decoded_qr(qr_data)
        if detected == self.verified_qr:
            self._note_verified_qr_seen()
            return

        if self._verified_last_seen_time is None:
            return
        if self._verified_not_visible_logged:
            return

        self._verified_lost_frames += 1
        if self._verified_lost_frames < QR_GONE_CONFIRM_FRAMES:
            return

        self._verified_visible = False
        self._verified_not_visible_logged = True
        gone_for = time.time() - self._verified_last_seen_time
        banner = (
            "==========================================\n"
            "QR CODE NOT VISIBLE\n"
            f"Verified QR     : {self.verified_qr}\n"
            f"Target type     : {self.expected_target_type}\n"
            f"Last QR seen    : {self._verified_last_seen_stamp}\n"
            f"Gone for        : {gone_for:.2f} s  "
            f"({self._verified_lost_frames} frames)\n"
            f"Seen frames     : {self._verified_seen_frames}\n"
            f"FSM             : {self.fsm_state}\n"
            "This is the last successful scan leaving the camera.\n"
            "==========================================")
        self.get_logger().info("\n" + banner)
        print(banner, flush=True)
        self._publish_qr_not_visible(True)

    # ------------------------------------------------------------------
    # Loopback observer callbacks (log only)
    # ------------------------------------------------------------------
    def target_type_callback(self, msg):
        self._log_ros_topic("/target_type", msg.data if msg else None)
        self.get_logger().info(
            f"Target type observed (log only, no mission activation): "
            f"{str(msg.data).strip() if msg and msg.data else ''}"
        )

    def target_qr_callback(self, msg):
        self._log_ros_topic("/target_qr", msg.data if msg else None)
        self.get_logger().info(
            f"Target QR observed (log only, no mission activation): "
            f"{str(msg.data).strip() if msg and msg.data else ''}"
        )

    def mission_turn_callback(self, msg):
        self._log_ros_topic("/mission/turn", msg.data if msg else None)
        self.get_logger().info(
            "EXIT CALLBACK: mission_turn_callback | REASON: Debug logging only"
        )

    # ==================================================================
    # MISSION FLOW
    # ==================================================================
    def _assign_mission(self, target_type, target_qr):
        """Step 1: Municipality assigns a mission."""
        self.get_logger().info(
            f"\n==========================================\n"
            f"MUNICIPALITY ASSIGNMENT\n"
            f"Target Type : {target_type}\n"
            f"Target QR   : {target_qr}\n"
            f"==========================================\n"
        )

        self.expected_target_type = target_type
        self.expected_qr = target_qr
        self.current_target_type_for_log = target_type

        if target_type == "PATIENT":
            self.mission_state = MISSION_PATIENT
            self.current_patient = target_qr
            self.current_patient_id = GOAL_TO_PATIENT_NUM.get(
                PATIENT_NAME_TO_GOAL.get(target_qr, "A"), 1)
            self.current_goal = PATIENT_NAME_TO_GOAL.get(target_qr, self.current_goal)
            self.assigned_hospital = None
        else:
            self.mission_state = MISSION_HOSPITAL
            self.assigned_hospital = target_qr
            self.assigned_hospital_id = GOAL_TO_HOSPITAL_NUM.get(
                HOSPITAL_NAME_TO_GOAL.get(target_qr, "X"), 4)
            self.current_goal = HOSPITAL_NAME_TO_GOAL.get(target_qr, self.current_goal)

        self.qr_verified = False
        self.verified_qr = None
        self._reset_verified_visibility()

        self._publish_mission_available(target_qr)
        self._publish_resume_line_following("RESUME")
        self.fsm_state = FSM_SEARCHING_QR

        self.get_logger().info(
            "==================================\n"
            "NEW MISSION (assigned)\n"
            f"Goal : {self.current_goal}\n"
            f"Expected QR : {self.expected_qr}\n"
            "qr_verified = False\n"
            f"FSM : {self.fsm_state}\n"
            f"Final mission (PARKED on safe_zone): {self._is_final_mission()}\n"
            "Searching for the matching QR only...\n"
            "=================================="
        )

        self._log_counters(f"After Assignment {target_qr}")

    def handle_qr_detection(self, qr_data):
        """Step 3: validate the detected QR against the expected QR."""
        if self.fsm_state != FSM_SEARCHING_QR:
            self.get_logger().info(
                f"QR ignored: {qr_data} - not searching (fsm={self.fsm_state})."
            )
            return

        if self.expected_qr is None:
            self.get_logger().info(
                f"QR ignored: {qr_data} - no expected QR assigned."
            )
            return

        detected = self._normalize_decoded_qr(qr_data)

        if detected != self.expected_qr:
            self.get_logger().info(
                f"Wrong {self.expected_target_type} Detected {detected} vs "
                f"Assigned {self.expected_qr} - ignored. Continuing Search..."
            )
            return

        if self.qr_verified and self.verified_qr == detected:
            self.get_logger().info(
                f"Duplicate QR ignored (already verified): {detected}"
            )
            return

        self.qr_verified = True
        self.verified_qr = detected
        self._note_verified_qr_seen()

        self.get_logger().info(
            f"\n----------------------------------\n"
            f"Correct {self.expected_target_type} Found via QR detection\n"
            f"Detected: {detected} Expected: {self.expected_qr}\n"
            f"QR VERIFIED : YES\n"
            f"Watching camera until this QR is no longer visible...\n"
            f"NEXT ACTION: Publish target and enter parking mode\n"
            f"----------------------------------\n"
        )

        self._publish_target_qr(detected)
        self._publish_target_type(self.expected_target_type)
        self._publish_qr_detection(detected)

        self.get_logger().info(
            f"FSM Transition: {self.fsm_state} -> {FSM_QR_VERIFIED}\n"
            "Waiting for /safe_zone..."
        )
        self.fsm_state = FSM_QR_VERIFIED

    def safe_zone_callback(self, message):
        """Step 5: /safe_zone=True sends the Municipality packet ONLY if the
        QR was verified.  Transition QR_VERIFIED -> SEND_PACKET.

        File 1 protocol: final hospital (goal Z) sends "PARKED" once
        instead of the goal letter. All other missions send A/B/C/X/Y.
        """
        self._log_ros_topic("/safe_zone", message.data if message else None)

        self.get_logger().info(
            f"\n==========================================\n"
            f"ENTER CALLBACK: safe_zone_callback\n"
            f"Bool Value: {message.data if message else None}\n"
            f"FSM: {self.fsm_state} Comm: {self.comm_state}\n"
            f"Expected QR: {self.expected_qr} Verified QR: {self.verified_qr}\n"
            f"qr_verified={self.qr_verified}\n"
            f"Goal: {self.current_goal} final={self._is_final_mission()} "
            f"parked_sent={self.parked_sent}\n"
            f"==========================================\n"
        )

        if message is None or not message.data:
            self.get_logger().info(
                "EXIT CALLBACK: safe_zone_callback | REASON: Bool False - "
                "safe zone not confirmed | NEXT ACTION: Ignore, continue waiting"
            )
            return

        if self.bonus_active:
            pass  # bonus park /safe_zone does not need a QR
        elif not self.qr_verified or self.verified_qr != self.expected_qr:
            self.get_logger().info(
                "EXIT CALLBACK: safe_zone_callback\n"
                "REASON: Safe zone ignored because QR has not been verified.\n"
                f"qr_verified={self.qr_verified} verified_qr={self.verified_qr} "
                f"expected_qr={self.expected_qr}\n"
                "NEXT ACTION: Continue searching, do not send Municipality packet"
            )
            return

        if self.comm_state == COMM_WAITING_ACK:
            self.get_logger().info(
                "EXIT CALLBACK: safe_zone_callback | REASON: Already waiting for "
                "ACK - do not resend | NEXT ACTION: Wait for ACK"
            )
            return

        self.get_logger().info(
            f"\n--------------------------------\n"
            f"SAFE ZONE RECEIVED\n"
            f"QR verified : YES\n"
            f"Expected QR : {self.expected_qr}\n"
            f"Verified QR : {self.verified_qr}\n"
            f"Goal        : {self.current_goal}\n"
            f"Sending Municipality Packet\n"
            f"--------------------------------\n"
        )

        # Bonus park finished: start / keep the PARKED loop running.
        if self.bonus_active:
            if self.comm_state != COMM_WAITING_ACK:
                self.send_mission_to_server("PARKED")
                self._last_parked_send = time.time()
            self.get_logger().info(
                "EXIT CALLBACK: safe_zone_callback | BONUS park "
                "— PARKED loop running")
            return

        # Normal legs (including Z) still send the goal letter.
        self.send_mission_to_server(self.current_goal)

        prev_fsm = self.fsm_state
        self.fsm_state = FSM_SEND_PACKET
        self.get_logger().info(
            f"FSM Transition: {prev_fsm} -> {self.fsm_state}\n"
            "Waiting ACK...\n"
        )
        self.get_logger().info(
            "EXIT CALLBACK: safe_zone_callback | REASON: Packet sent | NEXT ACTION: Wait for ACK"
        )

    # ==================================================================
    # Municipality payload parsing (packet format unchanged)
    # ==================================================================
    def _handle_mission_payload(self, payload):
        raw_payload = str(payload).strip()
        payload_upper = raw_payload.strip().upper()

        self.get_logger().info(
            f"\n==========================================\n"
            f"ENTER CALLBACK: _handle_mission_payload\n"
            f"Raw Payload: \"{raw_payload}\"\n"
            f"Normalized: \"{payload_upper}\"\n"
            f"Current FSM: {self.fsm_state}\n"
            f"==========================================\n"
        )

        extracted_hospital = self._extract_hospital_name(payload_upper)
        extracted_patient = self._extract_patient_name(payload_upper)

        if extracted_hospital:
            self._assign_mission("HOSPITAL", extracted_hospital)
            self._log_counters(f"After Hospital Assignment {extracted_hospital}")
            return

        if extracted_patient:
            self._assign_mission("PATIENT", extracted_patient)
            self._log_counters(f"After Patient Assignment {extracted_patient}")
            return

        if payload_upper not in VALID_MISSION_PAYLOADS:
            self.get_logger().warn(
                f"Unknown payload \"{raw_payload}\" - ignoring"
            )
            return

        if payload_upper in GOAL_TO_PATIENT_NAME:
            self._assign_mission("PATIENT", GOAL_TO_PATIENT_NAME[payload_upper])
            self._log_counters(f"After Patient Assignment {payload_upper} (legacy)")
            return

        if payload_upper in GOAL_TO_HOSPITAL_NAME:
            self._assign_mission("HOSPITAL", GOAL_TO_HOSPITAL_NAME[payload_upper])
            self._log_counters(f"After Hospital Assignment {payload_upper} (legacy)")
            return

        if payload_upper == "INVALID":
            if self.bonus_active and not self.bonus_ok:
                self.get_logger().info(
                    f"BONUS: server INVALID — keep sending PARKED every "
                    f"{PARKED_INTERVAL_SEC:.0f} s until OK "
                    f"(attempt {self._parked_attempts})")
                self.comm_state = COMM_IDLE
                self.pending_outgoing_msg = None
                return
            self.get_logger().info(
                f"Received INVALID from server - keeping current mission "
                f"(goal {self.current_goal}). Continuing search."
            )
            if not self._is_mission_active():
                self.fsm_state = FSM_SEARCHING_QR
            return

        if payload_upper == "OK":
            self.get_logger().info(
                f"MISSION FINISHED - Received OK -> {MISSION_COMPLETE}"
            )
            # A PARKED must actually have been sent for this OK to be the
            # parking confirmation. Otherwise this is the OK for the final
            # hospital leg, which arrives while bonus parking is only just
            # starting — halting the runner there would strand the buggy.
            parking_ok = (self._parked_attempts > 0 and
                          (self.bonus_active or self.bonus_ok))
            if parking_ok:
                self.bonus_ok = True
                self.bonus_active = False
                self.mission_state = MISSION_COMPLETE
                self._publish_resume_line_following("MISSION_COMPLETE")
                banner = (
                    "==========================================\n"
                    "BONUS: server OK — parking confirmed\n"
                    f"PARKED attempts: {self._parked_attempts}\n"
                    "RUN COMPLETE - BONUS PART FINISHED\n"
                    "==========================================")
                self.get_logger().info(banner)
                print(banner, flush=True)
            elif (self.bonus_active or
                  self.mission_leg_count >= TOTAL_MISSION_LEGS):
                # Final-leg OK while parking is under way. Do NOT send
                # MISSION_COMPLETE: it would stop the line follower dead
                # and the buggy would never reach the parking box.
                self.mission_state = MISSION_COMPLETE
                self.get_logger().info(
                    f"All {TOTAL_MISSION_LEGS} legs done — suppressing "
                    "MISSION_COMPLETE so the bonus parking run can finish "
                    f"(PARKED every {PARKED_INTERVAL_SEC:.0f} s until OK)")
            else:
                self.mission_state = MISSION_COMPLETE
                self._publish_resume_line_following("MISSION_COMPLETE")
            self.qr_verified = False
            self.verified_qr = None
            self.expected_qr = None
            self.expected_target_type = None
            self._reset_verified_visibility()
            self.fsm_state = FSM_SEARCHING_QR
            self._log_counters("After OK")
            return

    # ==================================================================
    # Communication layer (ACK / retry / UID - UNCHANGED)
    # ==================================================================
    def get_next_uid(self):
        uid = self.uid_counter
        self.uid_counter = (self.uid_counter + 1) % (UID_MAX + 1)
        return uid

    def _send_ack(self, uid):
        ts = self._get_timestamp_str()
        self.get_logger().info(
            f"\n----------------------------------\n"
            f"ENTER: _send_ack\n"
            f"Creating Municipality Packet\n"
            f"Packet Type: ACK\n"
            f"Source: {BUGGY_ID}\n"
            f"Destination: {SERVER_ID}\n"
            f"UID: {uid} (reused)\n"
            f"ACK: 1\n"
            f"Payload: \"\"\n"
            f"Timestamp: {ts}\n"
            f"----------------------------------\n"
        )

        ack_msg = ServerCommunication()
        setattr(ack_msg, SERVER_FIELD_SRC, BUGGY_ID)
        setattr(ack_msg, SERVER_FIELD_DEST, SERVER_ID)
        setattr(ack_msg, SERVER_FIELD_UID, uid)
        setattr(ack_msg, SERVER_FIELD_ACK, 1)
        setattr(ack_msg, SERVER_FIELD_MSG, "")

        self.get_logger().info("Publishing packet... Topic: /ServerCommunication (ACK)")

        try:
            self.publisher_server.publish(ack_msg)
            self.packets_sent += 1
            self.get_logger().info(
                f"Packet Published Successfully\n"
                f"Topic: /ServerCommunication\n"
                f"Timestamp: {self._get_timestamp_str()}\n"
            )
        except Exception as e:
            self.ack_failed += 1
            self.get_logger().error(f"Packet publish failed REASON: {e}")

    def send_mission_to_server(self, goal_letter):
        ts = self._get_timestamp_str()
        goal_letter = str(goal_letter).strip().upper()

        self.get_logger().info(
            f"\n----------------------------------\n"
            f"ENTER: send_mission_to_server\n"
            f"Goal: \"{goal_letter}\" comm_state={self.comm_state} fsm={self.fsm_state}\n"
            f"Timestamp: {ts}\n"
            f"----------------------------------\n"
        )

        if self.comm_state == COMM_WAITING_ACK:
            self.get_logger().warn(
                f"EXIT: send_mission_to_server REASON: Ignored because waiting for ACK"
            )
            self.duplicate_packets += 1
            return

        uid = self.get_next_uid()
        self.get_logger().info(
            f"Creating Municipality Packet\n"
            f"Packet Type: MISSION (ack=0)\n"
            f"Source: {BUGGY_ID}\n"
            f"Destination: {SERVER_ID}\n"
            f"UID: {uid} (new)\n"
            f"ACK: 0\n"
            f"Payload: \"{goal_letter}\"\n"
        )

        server_msg = ServerCommunication()
        setattr(server_msg, SERVER_FIELD_SRC, BUGGY_ID)
        setattr(server_msg, SERVER_FIELD_DEST, SERVER_ID)
        setattr(server_msg, SERVER_FIELD_UID, uid)
        setattr(server_msg, SERVER_FIELD_ACK, 0)
        setattr(server_msg, SERVER_FIELD_MSG, goal_letter)

        self.pending_outgoing_msg = server_msg
        self.pending_send_time = self.get_clock().now()
        self.retry_count = 0
        prev_state = self.comm_state
        self.comm_state = COMM_WAITING_ACK

        self.get_logger().info(
            f"Publishing packet... Topic: /ServerCommunication msg=\"{goal_letter}\" uid={uid}"
        )

        try:
            self.publisher_server.publish(server_msg)
            self.packets_sent += 1
            self.get_logger().info(
                f"Packet Published Successfully\n"
                f"Topic: /ServerCommunication\n"
                f"Timestamp: {self._get_timestamp_str()}\n"
                f"Comm State: {prev_state} -> {self.comm_state}\n"
            )
        except Exception as e:
            self.ack_failed += 1
            self.get_logger().error(f"Packet publish failed REASON: {e}")
            return

        self.get_logger().info(
            f"\n==========================================\n"
            f"Waiting for Municipality ACK...\n"
            f"Start timeout timer {ACK_TIMEOUT_SEC}s\n"
            f"UID: {uid} Payload: \"{goal_letter}\"\n"
            f"==========================================\n"
        )
        self.get_logger().info("EXIT: send_mission_to_server | NEXT ACTION: Wait ACK")

    def send_qr_to_server(self, qr_data):
        candidate = str(qr_data).strip().upper()
        if candidate in GOAL_TO_PATIENT_NAME or candidate in GOAL_TO_HOSPITAL_NAME:
            goal_to_send = candidate
        else:
            if candidate in PATIENT_NAME_TO_GOAL:
                goal_to_send = PATIENT_NAME_TO_GOAL[candidate]
            elif candidate in HOSPITAL_NAME_TO_GOAL:
                goal_to_send = HOSPITAL_NAME_TO_GOAL[candidate]
            else:
                pat = self._extract_patient_name(qr_data)
                hos = self._extract_hospital_name(qr_data)
                if pat and pat in PATIENT_NAME_TO_GOAL:
                    goal_to_send = PATIENT_NAME_TO_GOAL[pat]
                elif hos and hos in HOSPITAL_NAME_TO_GOAL:
                    goal_to_send = HOSPITAL_NAME_TO_GOAL[hos]
                else:
                    goal_to_send = self.current_goal
        self.send_mission_to_server(goal_to_send)

    def _communication_timeout_check(self):
        if self.comm_state != COMM_WAITING_ACK:
            return
        if self.pending_outgoing_msg is None or self.pending_send_time is None:
            return

        now = self.get_clock().now()
        elapsed = (now - self.pending_send_time).nanoseconds / 1e9

        if elapsed >= ACK_TIMEOUT_SEC:
            if self.retry_count < MAX_RETRIES:
                self.comm_state = COMM_RETRY
                uid = getattr(self.pending_outgoing_msg, SERVER_FIELD_UID)

                self.get_logger().warn(
                    f"FAILURE: ACK timeout No ACK for uid={uid} after {elapsed:.1f}s "
                    f"Retry {self.retry_count + 1}/{MAX_RETRIES}"
                )

                try:
                    self.publisher_server.publish(self.pending_outgoing_msg)
                    self.packets_sent += 1
                    self.retries_total += 1
                except Exception:
                    self.ack_failed += 1

                self.pending_send_time = now
                self.retry_count += 1
                self.comm_state = COMM_WAITING_ACK
            else:
                uid = getattr(self.pending_outgoing_msg, SERVER_FIELD_UID)
                msg = getattr(self.pending_outgoing_msg, SERVER_FIELD_MSG)
                self.get_logger().error(
                    f"Communication FAILURE! No ACK after {MAX_RETRIES} retries "
                    f"uid={uid} msg=\"{msg}\" Comm {self.comm_state} -> {COMM_FAILED}"
                )
                self.comm_state = COMM_FAILED
                self.ack_failed += 1

                self.pending_outgoing_msg = None
                self.pending_send_time = None
                self.retry_count = 0
                self.get_logger().info(
                    "Communication FAILURE\n"
                    "Current packet dropped.\n"
                    "Communication layer recovered.\n"
                    f"Comm State: {COMM_FAILED} -> {COMM_IDLE}\n"
                    "Waiting for next municipality assignment..."
                )
                self.comm_state = COMM_IDLE
                self.pending_send_time = None
                self._log_counters("After FAILURE")

    def server_communication_callback(self, message):
        ts = self._get_timestamp_str()
        self.get_logger().info(
            f"\n==========================================\n"
            f"ENTER CALLBACK: server_communication_callback\n"
            f"Topic: /ServerCommunication\n"
            f"Timestamp: {ts}\n"
            f"==========================================\n"
        )

        try:
            dest = getattr(message, SERVER_FIELD_DEST)
            src = getattr(message, SERVER_FIELD_SRC)
            ack = getattr(message, SERVER_FIELD_ACK)
            uid = getattr(message, SERVER_FIELD_UID)
            payload = getattr(message, SERVER_FIELD_MSG)
        except Exception as e:
            self.get_logger().error(f"Failed to parse ServerCommunication: {e}")
            return

        self._log_ros_topic(
            "/ServerCommunication",
            f"src={src} dest={dest} ack={ack} uid={uid} msg=\"{payload}\"")

        if src == BUGGY_ID and dest == SERVER_ID:
            self.get_logger().info(f"EXIT: Ignored self-published uid={uid}")
            return

        if src != SERVER_ID or dest != BUGGY_ID:
            self.get_logger().warn(
                f"ACK VALIDATION FAILED Checking UID uid={uid} Checking "
                f"Destination Expected src={SERVER_ID} dest={BUGGY_ID}, got "
                f"src={src} dest={dest} ACK INVALID"
            )
            return

        if ack == 1:
            self.get_logger().info(
                f"\n====================================\n"
                f"ACK RECEIVED\n"
                f"UID: {uid}\n"
                f"Source: {src}\n"
                f"Destination: {dest}\n"
                f"Status: ack=1\n"
                f"Message: \"{payload}\"\n"
                f"Timestamp: {self._get_timestamp_str()}\n"
                f"====================================\n"
            )
            self.get_logger().info(
                f"ENTER: ACK VALIDATION pending="
                f"{getattr(self.pending_outgoing_msg, SERVER_FIELD_UID, None) if self.pending_outgoing_msg else 'None'} "
                f"received={uid}"
            )

            if self.comm_state == COMM_WAITING_ACK and self.pending_outgoing_msg is not None:
                pending_uid = getattr(self.pending_outgoing_msg, SERVER_FIELD_UID)
                if uid == pending_uid:
                    self.get_logger().info(f"ACK VALID Reason: UID matches {pending_uid}")
                    prev = self.comm_state
                    self.comm_state = COMM_WAITING_SERVER_ASSIGNMENT
                    self.retry_count = 0
                    self.pending_send_time = None
                    self.ack_received += 1
                    self.get_logger().info(
                        f"ACK received from server uid={uid} matches pending msg="
                        f"\"{getattr(self.pending_outgoing_msg, SERVER_FIELD_MSG)}\" "
                        f"Comm {prev} -> {self.comm_state} ACK ONLY confirms reception"
                    )

                    pending_payload = str(
                        getattr(self.pending_outgoing_msg,
                                SERVER_FIELD_MSG, "") or "").strip().upper()
                    if pending_payload == "PARKED":
                        # Bonus PARKED ACK: do not resume; keep looping.
                        self.comm_state = COMM_IDLE
                        self.pending_outgoing_msg = None
                        self.get_logger().info(
                            f"BONUS: PARKED ACK — will send again in "
                            f"{PARKED_INTERVAL_SEC:.0f} s "
                            "until server OK (INVALID keeps going)")
                    elif self.fsm_state == FSM_SEND_PACKET:
                        self._note_leg_complete(
                            pending_payload or self.current_goal)
                        self.get_logger().info(
                            f"\n--------------------------------\n"
                            f"SAFE ZONE MISSION SUCCESS\n"
                            f"ACK Received - Publishing /resume_line_following\n"
                            f"Timestamp: {self._get_timestamp_str()}\n"
                            f"Legs: {self.mission_leg_count}/{TOTAL_MISSION_LEGS}\n"
                            f"--------------------------------\n"
                        )
                        if not self.bonus_active:
                            self._publish_resume_line_following("RESUME")
                        self.qr_verified = False
                        self.verified_qr = None
                        self.expected_qr = None
                        self.expected_target_type = None
                        self._reset_verified_visibility()
                        prev_fsm = self.fsm_state
                        self.fsm_state = FSM_SEARCHING_QR
                        self.get_logger().info(
                            f"FSM Transition: {prev_fsm} -> {self.fsm_state}\n"
                            "Waiting for the next Municipality assignment..."
                        )

                    self.get_logger().info(
                        "EXIT CALLBACK: server_communication_callback | REASON: Valid ACK"
                    )
                    self._log_counters("After ACK VALID")
                    return
                else:
                    self.get_logger().warn(
                        f"ACK INVALID UID mismatch received {uid} pending {pending_uid} "
                        "Failure: Duplicate ACK ignored"
                    )
                    self.duplicate_acks += 1
                    return
            elif self.comm_state == COMM_WAITING_SERVER_ASSIGNMENT:
                if self.pending_outgoing_msg is not None:
                    pending_uid = getattr(self.pending_outgoing_msg, SERVER_FIELD_UID)
                    if uid == pending_uid:
                        self.get_logger().info(
                            f"Duplicate ACK ignored uid={uid} while already in {self.comm_state}"
                        )
                        self.duplicate_acks += 1
                        return
                self.get_logger().info(
                    f"ACK received in {self.comm_state} uid={uid} duplicate"
                )
                self.duplicate_acks += 1
                return
            else:
                self.get_logger().info(
                    f"ACK received but comm_state={self.comm_state} not WAITING_ACK - duplicate/late"
                )
                if self.comm_state in (COMM_IDLE, COMM_FAILED):
                    self.get_logger().info(
                        "Late ACK ignored.\n"
                        "Reason:\n"
                        "Packet already marked failed."
                    )
                self.duplicate_acks += 1
                return

        if self.last_processed_server_uid is not None and uid == self.last_processed_server_uid:
            self.get_logger().warn(
                f"FAILURE: Duplicate packet ignored. Duplicate mission packet "
                f"uid={uid} msg=\"{payload}\" Send ACK again WITHOUT processing twice"
            )
            self.duplicate_packets += 1
            self._send_ack(uid)
            return

        self.get_logger().info(
            f"New mission packet from server: src={src} dest={dest} ack=0 "
            f"uid={uid} msg=\"{payload}\" Immediately ACKing"
        )
        self._send_ack(uid)
        self.last_processed_server_uid = uid
        self._handle_mission_payload(payload)
        self.get_logger().info("EXIT CALLBACK: server_communication_callback")

    # ==================================================================
    # Publish helpers (topic names unchanged — file 2)
    # ==================================================================
    def _publish_target_type(self, target_type):
        ts = self._get_timestamp_str()
        self.get_logger().info(
            f"\n----------------------------------\n"
            f"Publishing /target_type: {target_type}\n"
            f"Timestamp: {ts}\n"
            f"----------------------------------\n"
        )
        msg = String()
        msg.data = target_type
        self.publisher_target_type.publish(msg)
        self.get_logger().info(
            f"Packet Published Successfully\n"
            f"Topic: /target_type\n"
            f"Timestamp: {self._get_timestamp_str()}\n"
        )

    def _publish_target_qr(self, target_qr):
        ts = self._get_timestamp_str()
        self.get_logger().info(
            f"\n----------------------------------\n"
            f"Publishing /target_qr: {target_qr}\n"
            f"Timestamp: {ts}\n"
            f"----------------------------------\n"
        )
        msg = String()
        msg.data = target_qr
        self.publisher_target_qr.publish(msg)
        self.get_logger().info(
            f"Packet Published Successfully\n"
            f"Topic: /target_qr\n"
            f"Timestamp: {self._get_timestamp_str()}\n"
        )

    def _publish_qr_detection(self, qr_data):
        ts = self._get_timestamp_str()
        self.get_logger().info(
            f"\n----------------------------------\n"
            f"Publishing /qr_detection: {qr_data}\n"
            f"Timestamp: {ts}\n"
            f"----------------------------------\n"
        )
        msg = String()
        msg.data = qr_data
        self.publisher_qr.publish(msg)
        self.get_logger().info(
            f"Packet Published Successfully\n"
            f"Topic: /qr_detection\n"
            f"Timestamp: {self._get_timestamp_str()}\n"
        )

    def _publish_resume_line_following(self, payload="RESUME"):
        ts = self._get_timestamp_str()
        self.get_logger().info(
            f"\n----------------------------------\n"
            f"Publishing /resume_line_following: {payload}\n"
            f"Timestamp: {ts}\n"
            f"----------------------------------\n"
        )
        msg = String()
        msg.data = payload
        self.publisher_resume.publish(msg)
        self.get_logger().info(
            f"Packet Published Successfully\n"
            f"Topic: /resume_line_following\n"
            f"Timestamp: {self._get_timestamp_str()}\n"
        )

    def _publish_mission_available(self, target_qr):
        msg = String()
        msg.data = target_qr
        self.publisher_mission_available.publish(msg)
        self.get_logger().info(f"Published /mission/available: {target_qr}")

    # ==================================================================
    # Camera / QR vision pipeline
    # ==================================================================
    def camera_image_callback(self, message):
        """Decode frame -> detect QR -> overlay -> validate -> visibility."""
        try:
            np_arr = np.frombuffer(message.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            self.get_logger().error(f"Failed to decode compressed image: {e}")
            return

        if image is None or image.size == 0:
            return

        qr_data, bbox = self.detect_qr_code(image)

        if bbox is not None:
            self.draw_qr_overlay(image, bbox, qr_data)

        if qr_data:
            self.handle_qr_detection(qr_data)

        self._update_verified_qr_visibility(qr_data)

        try:
            cv2.imshow("QR Detector", image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
        except Exception as e:
            self.get_logger().debug(f"OpenCV visualization failed: {e}")

    def image_variants(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        variants = [(gray, 1.0)]
        variants.append((self.clahe.apply(gray), 1.0))
        variants.append((
            cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC),
            0.5))

        return variants

    def detect_qr_code(self, image):
        variants = self.image_variants(image)

        for detector in (self.detect_with_pyzbar, self.detect_with_opencv):
            data, points = detector(variants)
            if data:
                return data, points

        return None, None

    def detect_with_pyzbar(self, variants):
        if not PYZBAR_AVAILABLE:
            return None, None

        for img, scale in variants:
            try:
                decoded_objects = pyzbar.decode(img)
            except Exception:
                continue

            for obj in decoded_objects:
                try:
                    data = obj.data.decode('utf-8').strip()
                except Exception:
                    continue
                if not data:
                    continue

                if obj.polygon and len(obj.polygon) >= 4:
                    points = np.array([[p.x, p.y] for p in obj.polygon],
                                      dtype=np.float32)
                else:
                    x, y, w, h = obj.rect
                    points = np.array(
                        [[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                        dtype=np.float32)

                points *= scale
                return data, points

        return None, None

    def detect_with_opencv(self, variants):
        for img, scale in variants:
            try:
                data, bbox, _ = self.qr_detector.detectAndDecode(img)
            except Exception:
                continue

            if bbox is None or not data or not data.strip():
                continue

            points = bbox.reshape(-1, 2).astype(np.float32)
            points *= scale
            return data.strip(), points

        return None, None

    def draw_qr_overlay(self, image, bbox, qr_data):
        try:
            pts = bbox.reshape(-1, 2).astype(int)
            for i in range(len(pts)):
                cv2.line(image, tuple(pts[i]),
                         tuple(pts[(i + 1) % len(pts)]), (0, 255, 0), 2)
            if qr_data:
                cv2.putText(
                    image, qr_data,
                    (pts[0][0], max(pts[0][1] - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except Exception:
            pass

    def destroy_node(self):
        self.get_logger().info("ENTER: destroy_node")
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        self._log_counters("At Shutdown")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = QRDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
