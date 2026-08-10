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

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from collections import deque
import cv2
import numpy as np

# ============================================================================
# FAST CLASSICAL-CV SIGN DETECTOR
# ============================================================================
# No ML / DL / OCR.
# Pipeline per board:
#   green board detection -> perspective warp -> equal 6-cell split ->
#   dual-window white arrow extraction -> geometry classification.
#
# Multi-board support is kept: every detected mid-range board is processed.
# The ROS node chooses the highest-confidence result for the current goal.
# ============================================================================

CANONICAL_W = 600
CANONICAL_H = 120
LETTER_ORDER = ["A", "B", "C", "X", "Y", "Z"]
CELL_LABELS = LETTER_ORDER

# Mid-range gate. Increase these if it still detects too early/far away.
MIN_DETECT_BOARD_WIDTH = 180
MIN_DETECT_BOARD_HEIGHT = 48
MIN_BOARD_AREA = 1200

# Minimum board width to TRUST a read for locking. The board is still DETECTED
# early (MIN_DETECT_BOARD_WIDTH) for tracking, but a vote is only cast once the
# board is close enough (this width) that the arrow is reliably readable.
# Far / mid-range reads are degraded and wrong, and they cause the
# LEFT<->RIGHT<->STRAIGHT oscillation + premature locks (e.g. locking a
# transitional RIGHT before the true STRAIGHT appears closer up). Gating votes
# by closeness removes them. TUNE PER COURSE: raise if it locks a wrong /
# transitional direction (wait for closer), lower if it abstains too much.
MIN_LOCK_BOARD_WIDTH = 220

# When multiple boards are visible, do not choose the highest-confidence arrow
# from any board in the image. That can pick a side/old board. Choose the board
# that is in front of the robot: near image center and reasonably large.
BOARD_CENTER_GATE = 0.65       # 0=center, 1=edge. Reject boards beyond this.
BOARD_CENTER_WEIGHT = 2.0      # higher = stronger preference for centered board

# Board exit reset. After mission lock, the node waits until the board is gone
# for this many frames, then unlocks and searches again for the SAME current
# goal. /mission/available (from the QR Detector at assignment time) changes
# the goal at any time.
EXIT_MISSING_FRAMES_MAX = 12

# Do not tolerate missed/low-confidence frames for locking. A single bad frame
# resets the streak, preventing false RIGHT/STRAIGHT locks while approaching.
SKIP_MISSING_FRAMES_MAX = 0

GREEN_HSV_LOW = np.array([35, 35, 35])
GREEN_HSV_HIGH = np.array([95, 255, 255])

# Arrow extraction / classification tuning.
#
# These signs place the arrow at two different vertical positions:
#   * lower cell        -> ARROW_Y0_RATIO .. ARROW_Y1_RATIO
#   * upper-middle cell -> ARROW_UPPER_Y0 .. ARROW_UPPER_Y1
# read_arrow_direction() tries BOTH windows and keeps the confident read. A
# single fixed window matched only one layout (39%); the dual window hits both.
ARROW_Y0_RATIO = 0.58
ARROW_Y1_RATIO = 0.96
ARROW_UPPER_Y0 = 0.30
ARROW_UPPER_Y1 = 0.55
ARROW_X_MARGIN_RATIO = 0.07
WHITE_HSV_LOW = np.array([0, 0, 115])
WHITE_HSV_HIGH = np.array([180, 125, 255])

# A single arrow cannot fill most of the ROI. If the selected white blob covers
# more than this fraction of the ROI, the cell is contaminated (white margin /
# glare / mis-warp) and its direction is unreliable -> return None instead of a
# flippy low-confidence call.
ARROW_FILL_MAX = 0.45

# Quality gate: only trust a direction when the arrow blob is substantial.
# Below these, the arrow is too small / degraded (far range, blur) to read
# reliably, so return None ("in doubt, leave it") instead of a confident wrong
# call. Calibrated so every clean sample arrow passes (bh>=10, pixels>=216);
# degraded far-range arrows (bh 7-9, ~140 px) are rejected.
ARROW_MIN_HEIGHT = 10
ARROW_MIN_PIXELS = 200

# Straight must be REALLY vertical/narrow. Earlier value 1.45 caused broken
# LEFT/RIGHT arrow fragments to be called STRAIGHT at mid range.
STRAIGHT_ASPECT_MAX = 1.05
# Treat moderate-width blobs as horizontal arrows; if centroid is weak they
# will be skipped instead of guessed.
HORIZONTAL_ASPECT_MIN = 1.20
DIR_ASYM_DEAD_ZONE = 0.10
DIR_ASYM_STRONG = 0.25
# Horizontal LEFT/RIGHT decision. Centroid is more stable than row-extents at
# mid/far range. Negative centroid offset = LEFT, positive = RIGHT.
CENTROID_DEAD_ZONE = 0.025
CENTROID_STRONG = 0.055
# Do not accept weak STRAIGHT reads. False A/B straight mistakes were
# low-confidence (~0.82). Real straight arrows in samples score >0.90, so weak
# STRAIGHT reads are ignored.
STRAIGHT_ACCEPT_CONF = 0.90
DEFAULT_CONFIDENCE_THRESHOLD = 0.90
DEFAULT_REQUIRED_CONSECUTIVE = 5

# Temporal lock: a SLIDING WINDOW of recent reads decides the lock, not a
# running total. Early far-range wrong reads fall out of the window so the
# close-range correct read can win. A lock needs a FULL window AND a clear
# supermajority; if the approach oscillates ~50/50 no direction wins, so it
# abstains (no wrong lock). VOTE_WINDOW = number of recent votes kept;
# VOTE_SUPERMAJORITY = winner's required share of the window.
VOTE_WINDOW = 8
VOTE_SUPERMAJORITY = 0.66

# Mission mapping for /mission/available (replaces Municipality Server dest parsing)
PATIENT_QR_TO_GOAL = {
    "PATIENT_1": "A",
    "PATIENT_2": "B",
    "PATIENT_3": "C",
}

HOSPITAL_QR_TO_GOAL = {
    "HOSPITAL_1": "X",
    "HOSPITAL_2": "Y",
    "HOSPITAL_3": "Z",
}

TARGET_QR_TO_GOAL = {**PATIENT_QR_TO_GOAL, **HOSPITAL_QR_TO_GOAL}


def order_points(pts):
    pts = pts.reshape(4, 2).astype(np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    d = np.diff(pts, axis=1).flatten()
    ordered[1] = pts[np.argmin(d)]
    ordered[3] = pts[np.argmax(d)]
    return ordered


def get_board_quad(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    if len(approx) == 4:
        quad = approx.reshape(4, 2).astype(np.float32)
    else:
        quad = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
    return order_points(quad)


def correct_perspective(frame, quad, target_w=CANONICAL_W, target_h=CANONICAL_H):
    dst = np.array(
        [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(frame, matrix, (target_w, target_h), flags=cv2.INTER_LINEAR)


def green_board_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, GREEN_HSV_LOW, GREEN_HSV_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def detect_boards(frame):
    """Return all mid-range green boards, sorted largest first."""
    mask = green_board_mask(frame)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boards = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_BOARD_AREA:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        if w < MIN_DETECT_BOARD_WIDTH or h < MIN_DETECT_BOARD_HEIGHT:
            continue
        aspect = w / float(h + 1e-6)
        if aspect < 2.0:
            continue
        quad = get_board_quad(contour)
        warped = correct_perspective(frame, quad)
        boards.append({
            "contour": contour,
            "quad": quad,
            "warped": warped,
            "bbox": (x, y, x + w, y + h),
            "area": area,
        })
    boards.sort(key=lambda b: -b["area"])
    return boards


def detect_board(frame):
    """Compatibility helper: returns the largest board ROI and bbox."""
    boards = detect_boards(frame)
    if not boards:
        return None, None
    board = boards[0]
    return board["warped"], board["bbox"]


def split_cells(warped, n_cells=6):
    h, w = warped.shape[:2]
    cell_w = w // n_cells
    cells = []
    for i in range(n_cells):
        x0 = i * cell_w
        x1 = (i + 1) * cell_w if i < n_cells - 1 else w
        cells.append((warped[:, x0:x1], (x0, x1)))
    return cells


def arrow_mask_from_cell(cell, y0_ratio=ARROW_Y0_RATIO, y1_ratio=ARROW_Y1_RATIO):
    ch, cw = cell.shape[:2]
    y0 = int(ch * y0_ratio)
    y1 = int(ch * y1_ratio)
    xm = max(2, int(cw * ARROW_X_MARGIN_RATIO))
    roi = cell[y0:y1, xm:cw - xm]
    if roi.size == 0:
        return None, None, None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, WHITE_HSV_LOW, WHITE_HSV_HIGH)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask, roi, (xm, y0)


def select_arrow_component(mask):
    """Return a cleaned UNION of arrow components, not just the largest blob.

    Straight/up arrows can split into two pieces at mid range: triangular head
    and vertical shaft. Picking only the largest piece makes B/Z look like a
    horizontal LEFT/RIGHT arrow. So we keep all reasonable arrow pieces and only
    reject tiny noise and edge divider artifacts.
    """
    h, w = mask.shape[:2]
    roi_area = h * w
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    selected = np.zeros_like(mask)
    min_area = max(10, int(roi_area * 0.003))
    kept_any = False
    best_score = 0.0
    best_id = None
    for cid in range(1, n_labels):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        x, y, bw, bh = stats[cid, 0:4]
        cx, cy = centroids[cid]
        if area < min_area:
            continue
        # Reject tiny specks.
        if bw < 3 or bh < 3:
            continue
        near_edge = (x <= 2) or (x + bw >= w - 2)
        # Reject small edge specks/fragments that expand the bbox and ruin
        # STRAIGHT detection, especially in Z.
        if near_edge and area < 45:
            continue
        # Reject vertical divider fragments. Dividers are narrow and almost
        # full ROI height. A true STRAIGHT shaft is narrow too, but it is not
        # full height and is usually connected to the arrow head.
        if bw <= 10 and bh > 0.72 * h:
            continue
        if near_edge and bw <= 8 and bh > 0.30 * h:
            continue
        # Reject thin horizontal divider/border stripes: they span most of the
        # ROI width but are very short. These are letter/arrow separator lines
        # caught at far or tilted range (else classified as a false horizontal
        # arrow). A real horizontal arrow has a tall triangular head, so it is
        # never a thin full-width band.
        if bw >= 0.80 * w and bh <= 0.30 * h:
            continue
        # Reject full-height vertical fragments (letter strokes / borders that
        # bleed into the ROI at tilted or far range). A real arrow never spans
        # the entire ROI height (top AND bottom edges). Width-bounded so genuine
        # straight-arrow heads (wider, with margin) are not affected.
        if y <= 1 and (y + bh) >= (h - 1) and bw <= 14:
            continue
        # Reject sparse "frame" noise that spans the whole ROI (touches all four
        # edges) from heavy JPEG/tilt. Dense floods are caught later by the fill
        # guard; this catches low-area full-span noise. A real arrow never
        # touches all four ROI edges at once.
        if x <= 1 and (x + bw) >= (w - 1) and y <= 1 and (y + bh) >= (h - 1):
            continue
        # Reject high fragments; after the lower crop, real arrow pixels are
        # still in the middle/lower part of the ROI. This prevents letter
        # pieces, especially the vertical part of 'A', from being classified
        # as a STRAIGHT arrow at mid range.
        if cy < 0.18 * h:
            continue
        selected[labels == cid] = 255
        kept_any = True
        bbox_area = float(bw * bh)
        extent = area / max(bbox_area, 1.0)
        cy_bonus = 0.5 + cy / float(h + 1e-6)
        score = area * extent * cy_bonus
        if score > best_score:
            best_score = score
            best_id = cid
    if kept_any:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, kernel, iterations=1)
        return selected
    # Fallback: keep best component if everything was filtered too hard.
    if best_id is not None:
        selected[labels == best_id] = 255
        return selected
    return None


def classify_arrow_mask(mask):
    ys, xs = np.where(mask > 0)
    if xs.size < 20 or ys.size < 20:
        return None, 0.0
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw = x1 - x0 + 1
    bh = y1 - y0 + 1
    if bw < 7 or bh < ARROW_MIN_HEIGHT:
        return None, 0.0
    aspect = bw / float(bh + 1e-6)
    pixels = int(xs.size)
    # Quality gate: a reliable arrow has enough mass. Tiny/degraded blobs (far
    # range, blur) score high confidence but are wrong -> reject ("in doubt,
    # leave it") so a direction is only emitted when the arrow is substantial.
    if pixels < ARROW_MIN_PIXELS:
        return None, 0.0
    sub = (mask[y0:y1 + 1, x0:x1 + 1] > 0).astype(np.uint8)
    col_counts = sub.sum(axis=0).astype(np.float32)
    row_counts = sub.sum(axis=1).astype(np.float32)
    max_col_fill = float(col_counts.max()) / float(max(bh, 1))
    max_row_fill = float(row_counts.max()) / float(max(bw, 1))

    # Reject disconnected fragments unioned into a false shape. If a large
    # fraction of columns in the bbox are empty, the selected mask is really
    # two or more separated blobs (letter pieces / partial arrow bits caught at
    # far or tilted range), not one connected arrow.
    if bw > 0 and int(np.count_nonzero(col_counts)) < 0.55 * bw:
        return None, 0.0

    # ------------------------------------------------------------------
    # STRAIGHT / UP detector.
    # A straight arrow has a central vertical shaft + head. The important
    # signature is a tall vertical column. This is checked BEFORE left/right
    # so B/Z do not become LEFT/RIGHT when head/shaft split or blur.
    # ------------------------------------------------------------------
    vertical_like = (
        pixels >= 35 and
        bh >= 14 and
        aspect <= 1.35 and
        max_col_fill >= 0.45
    )
    very_vertical = (
        pixels >= 35 and
        bh >= 18 and
        aspect <= 1.10
    )
    if vertical_like or very_vertical:
        # confidence from vertical strength and narrowness
        aspect_conf = np.clip((1.35 - aspect) / max(1.35 - 0.55, 1e-6), 0.0, 1.0)
        col_conf = np.clip(max_col_fill / 0.75, 0.0, 1.0)
        conf = max(0.82, min(1.0, 0.55 * aspect_conf + 0.45 * col_conf))
        if conf < STRAIGHT_ACCEPT_CONF:
            return None, 0.0
        return "STRAIGHT", float(conf)

    # LEFT/RIGHT requires a clearly horizontal arrow. If it is not clearly
    # vertical and not clearly horizontal, skip instead of guessing.
    horizontal_like = (
        aspect >= HORIZONTAL_ASPECT_MIN and
        max_row_fill >= 0.35 and
        bw >= 14
    )
    if not horizontal_like:
        return None, 0.0
    # Centroid relative to bbox center is stable for left/right on this board.
    bbox_center_x = 0.5 * (x0 + x1)
    centroid_x = float(xs.mean())
    centroid_offset = (centroid_x - bbox_center_x) / float(max(bw, 1))
    if centroid_offset < -CENTROID_DEAD_ZONE:
        direction = "LEFT"
    elif centroid_offset > CENTROID_DEAD_ZONE:
        direction = "RIGHT"
    else:
        return None, 0.0
    conf = max(0.82, min(1.0, abs(centroid_offset) / CENTROID_STRONG))
    return direction, float(conf)


def read_arrow_direction(cell, debug=False):
    """Read one cell's arrow, trying both vertical layouts and keeping the
    most confident classification. Contaminated (flooded) ROIs are rejected.

    Returns (direction, confidence); direction is None if no reliable read.
    """
    best = (None, 0.0)
    best_artifacts = None
    for y0_ratio, y1_ratio in (
        (ARROW_UPPER_Y0, ARROW_UPPER_Y1),   # upper-middle layout
        (ARROW_Y0_RATIO, ARROW_Y1_RATIO),   # lower layout
    ):
        mask, roi, offset = arrow_mask_from_cell(cell, y0_ratio, y1_ratio)
        if mask is None:
            continue
        arrow = select_arrow_component(mask)
        if arrow is None:
            continue
        # Flood-guard: one arrow cannot fill most of the ROI -> contaminated.
        if arrow.size and (arrow.sum() / 255.0) / arrow.size > ARROW_FILL_MAX:
            continue
        direction, confidence = classify_arrow_mask(arrow)
        if direction is not None and confidence > best[1]:
            best = (direction, confidence)
            best_artifacts = (roi, mask, arrow)
    if debug and best_artifacts is not None:
        roi, mask, arrow = best_artifacts
        cv2.imshow("arrow_roi", roi)
        cv2.imshow("arrow_mask", mask)
        cv2.imshow("arrow_selected", arrow)
        cv2.waitKey(1)
    return best


def read_single_board(warped):
    """Read one already-warped board: {letter: (direction, confidence)}."""
    cells = split_cells(warped, len(LETTER_ORDER))
    results = {}
    for i, (cell, _) in enumerate(cells):
        letter = LETTER_ORDER[i]
        direction, confidence = read_arrow_direction(cell)
        results[letter] = (direction, confidence)
    return results


def read_boards(frame):
    """
    Process every detected board.
    Returns a list:
        [{"bbox": (...), "warped": image, "results": {letter: (dir, conf)}}]
    """
    boards = detect_boards(frame)
    output = []
    for board in boards:
        output.append({
            "bbox": board["bbox"],
            "warped": board["warped"],
            "area": board["area"],
            "results": read_single_board(board["warped"]),
        })
    return output


def read_board(frame):
    """
    Compatibility helper.
    Returns {letter: (direction, confidence)} using the highest confidence
    across all visible boards.
    """
    board_reads = read_boards(frame)
    results = {}
    for board in board_reads:
        for letter, result in board["results"].items():
            direction, confidence = result
            prev = results.get(letter)
            if prev is None or confidence > prev[1]:
                results[letter] = result
    return results


def board_target_score(board, frame_shape):
    """Score board for mission decision. Prefer centered forward board."""
    img_h, img_w = frame_shape[:2]
    x0, y0, x1, y1 = board["bbox"]
    bw = x1 - x0
    bh = y1 - y0
    cx = 0.5 * (x0 + x1)
    center_norm = abs(cx - 0.5 * img_w) / max(0.5 * img_w, 1.0)
    if center_norm > BOARD_CENTER_GATE:
        return -1.0
    center_score = max(0.0, 1.0 - center_norm) ** BOARD_CENTER_WEIGHT
    size_score = float(bw * bh)
    return size_score * center_score


def select_target_board(board_reads, frame_shape):
    """Select the single board that is most likely ahead of the robot."""
    best = None
    best_score = -1.0
    for board in board_reads:
        score = board_target_score(board, frame_shape)
        if score > best_score:
            best_score = score
            best = board
    if best_score < 0.0:
        return None
    return best


# ============================================================================
# ROS2 NODE - SIMPLIFIED COMMUNICATION: NO MUNICIPALITY SERVER
# ============================================================================
class ObjectRecognizer(Node):
    """
    ROS2 node:
      - subscribes camera image
      - subscribes /mission/available for immediate goal updates (Purpose 1:
        Municipality -> QR Detector -> Object Recognizer).  The goal updates
        at assignment time, before the QR is verified.  It does NOT depend on
        /target_qr (that topic is the QR Detector -> Line Follower path,
        Purpose 2, published only after QR match).
      - processes every detected board
      - publishes locked turn direction on /mission/turn
      - after the board disappears, resets and searches again for current goal
    """

    def __init__(self):
        super().__init__('object_recognizer')

        self.declare_parameter('goal_letter', 'A')
        self.declare_parameter('min_board_width', MIN_DETECT_BOARD_WIDTH)
        self.declare_parameter('min_board_height', MIN_DETECT_BOARD_HEIGHT)
        self.declare_parameter('confidence_threshold', DEFAULT_CONFIDENCE_THRESHOLD)
        self.declare_parameter('required_consecutive', DEFAULT_REQUIRED_CONSECUTIVE)
        self.declare_parameter('exit_missing_frames', EXIT_MISSING_FRAMES_MAX)
        self.declare_parameter('min_lock_board_width', MIN_LOCK_BOARD_WIDTH)

        initial_goal = self.get_parameter('goal_letter').get_parameter_value().string_value.upper()
        self.goal_letter = initial_goal if initial_goal in LETTER_ORDER else 'A'
        self.goal_valid = True
        self._last_param_goal = self.goal_letter

        self.mission_locked = False
        self.current_mission = None
        self.missing_frames = 0

        # Temporal vote window for the current approach. A sliding window (not a
        # running total) so early far-range wrong reads fall off and the
        # close-range correct read can win; a lock needs a full window + a
        # clear supermajority.
        self.vote_window = deque(maxlen=VOTE_WINDOW)

        # Mission target cache (replaces Municipality Server).
        # latest_target_qr holds the CANONICAL target (from /mission/available)
        # of the CURRENT mission.  Initialized to None so the first real
        # mission is never mistaken for a duplicate.
        self.latest_target_type = "PATIENT"
        self.latest_target_qr = None

        self.subscription_camera = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.camera_image_callback,
            10)

        # Purpose 1 (Municipality -> QR Detector -> Object Recognizer):
        # every new mission is received immediately after Municipality
        # assignment via /mission/available.  The QR Detector publishes this
        # topic at assignment time with the target QR payload (e.g. PATIENT_2).
        # The detector does NOT depend on /target_qr for goal updates -
        # /target_qr (Purpose 2) is only published after the QR is verified and
        # is consumed by the Line Follower, not by this node.
        self.subscription_mission_available = self.create_subscription(
            String,
            '/mission/available',
            self.mission_available_callback,
            10)

        self.publisher_turn = self.create_publisher(
            String,
            '/mission/turn',
            10)

        self.get_logger().info(
            f"Object Recognizer started. Multi-board enabled. Active goal = {self.goal_letter}. "
            f"Listening to /mission/available for immediate mission updates (Municipality Server removed).")

    def _reset_detection_state(self):
        """Full detector-state reset for a brand-new mission.

        Clears every counter/streak/lock/cached value that belongs to the
        previous mission so the detector can never continue searching for an
        old goal.
        """
        self.current_mission = None
        self.missing_frames = 0
        self.mission_locked = False
        self.goal_valid = True
        self.vote_window.clear()

    def _set_goal(self, new_goal, source):
        if new_goal is None or new_goal not in LETTER_ORDER:
            return False
        if new_goal != self.goal_letter or not self.goal_valid:
            self.goal_letter = new_goal
            self.goal_valid = True
            self.mission_locked = False
            self._reset_detection_state()
            self.get_logger().info(
                f"Goal set to '{new_goal}' from {source}. Detector reset.")
            return True
        return False

    def _reset_after_board_exit(self):
        """After completing one board, keep same goal and search again."""
        current_goal = self.goal_letter
        self.goal_valid = True
        self.mission_locked = False
        self._reset_detection_state()
        self.get_logger().info(
            f"Board exited. Continuing with current goal '{current_goal}'. Waiting for next board or /mission/available update.")

    # ------------------------------------------------------------------
    # PURPOSE-1 COMMUNICATION: Municipality -> QR Detector -> Object Recognizer
    # ------------------------------------------------------------------
    def mission_available_callback(self, msg):
        """
        Receive every new mission from the QR Detector immediately after a
        Municipality assignment, via /mission/available.

        This is the ONLY goal-update path for the Object Recognizer.  It runs
        at ASSIGNMENT time, BEFORE the QR is verified, so the detector always
        searches for the newest goal without waiting for QR verification and
        without a node restart.

        The detector does NOT depend on /target_qr for goal updates.  /target_qr
        (Purpose 2: QR Detector -> Line Follower) is published only after the
        QR has actually been matched and is consumed by the Line Follower.

        Mapping (kept exactly):
            PATIENT_1->A, PATIENT_2->B, PATIENT_3->C,
            HOSPITAL_1->X, HOSPITAL_2->Y, HOSPITAL_3->Z

        A duplicate mission (same target as the current one) is ignored -
        no reset, no stale goal.
        """
        if msg is None or not msg.data:
            return

        raw_qr = msg.data.strip()
        qr_upper = raw_qr.strip().upper()

        # Normalize: extract PATIENT_* or HOSPITAL_* if wrapped in {LOC: ...} or similar
        normalized_qr = qr_upper
        for known in TARGET_QR_TO_GOAL.keys():
            if known in qr_upper:
                normalized_qr = known
                break

        if normalized_qr not in TARGET_QR_TO_GOAL:
            self.get_logger().info(
                f"Ignoring /mission/available: '{raw_qr}' - not in mapping "
                f"{list(TARGET_QR_TO_GOAL.keys())}")
            return

        mapped_goal = TARGET_QR_TO_GOAL[normalized_qr]

        # Infer target type for logging.
        if "PATIENT" in normalized_qr:
            target_type = "PATIENT"
        elif "HOSPITAL" in normalized_qr:
            target_type = "HOSPITAL"
        else:
            target_type = self.latest_target_type

        # Duplicate handling.  Identical target to the current mission -> do
        # nothing, do NOT reset, do NOT clear counters.  Only log.
        if self.latest_target_qr is not None and self.latest_target_qr == normalized_qr:
            self.get_logger().info(
                "Duplicate mission ignored.\n"
                f"Target Type : {target_type}\n"
                f"Target QR   : {normalized_qr}\n"
                f"Current Goal: {self.goal_letter}\n"
                "No state reset performed."
            )
            return

        # NEW MISSION: keep the old goal only for logging, then fully reset.
        old_goal = self.goal_letter

        # Discard the previous mission completely.
        self._reset_detection_state()   # clears current_mission, missing_frames,
                                        # mission_locked, goal_valid, vote tally

        # Assign the new mapped goal.
        self.goal_letter = mapped_goal
        self.goal_valid = True
        self._last_param_goal = mapped_goal

        # Synchronize every cached mission variable to the new mission.
        self.latest_target_qr = normalized_qr
        self.latest_target_type = target_type

        # Synchronize the ROS parameter that stores the goal.
        try:
            self.set_parameters([
                Parameter('goal_letter', Parameter.Type.STRING, mapped_goal)
            ])
        except Exception:
            pass

        # Detailed log per required format.
        self.get_logger().info(
            "\n========================================\n"
            "NEW MISSION RECEIVED (from /mission/available)\n"
            "\n"
            f"Target Type : {target_type}\n"
            f"Target QR   : {normalized_qr}\n"
            "\n"
            f"Old Goal    : {old_goal}\n"
            f"New Goal    : {mapped_goal}\n"
            "\n"
            f"Mission Reset : YES\n"
            f"Mission Lock  : CLEARED\n"
            f"Counters      : RESET\n"
            "\n"
            f"Searching for Goal {mapped_goal}...\n"
            "========================================\n"
        )

        if old_goal != mapped_goal:
            self.get_logger().info(
                f"Goal changed: {old_goal} -> {mapped_goal} from {raw_qr} (type {target_type}). "
                f"Previous mission lock cleared, vote tally reset, board exit counter reset.")
        else:
            self.get_logger().info(
                f"Goal {mapped_goal} reconfirmed from {raw_qr}. Detector reset for fresh search.")

    def _check_parameter_goal(self):
        """
        Parameter support is kept, but it only reacts to actual parameter
        changes. The current goal persists after board exit unless
        /mission/available or parameter explicitly changes it.
        """
        param_goal = self.get_parameter('goal_letter').get_parameter_value().string_value.upper()
        if param_goal not in LETTER_ORDER:
            return
        if param_goal != self._last_param_goal:
            self._last_param_goal = param_goal
            self._set_goal(param_goal, 'parameter')

    def _handle_locked_wait_for_exit(self, image):
        boards = detect_boards(image)
        exit_missing_frames = self.get_parameter('exit_missing_frames').get_parameter_value().integer_value
        if not boards:
            self.missing_frames += 1
            self.get_logger().info(
                f"Board absent after mission lock ({self.missing_frames}/{exit_missing_frames})",
                throttle_duration_sec=1.0)
            if self.missing_frames >= exit_missing_frames:
                self._reset_after_board_exit()
        else:
            self.missing_frames = 0

    def camera_image_callback(self, message):
        np_arr = np.frombuffer(message.data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if image is None:
            return

        # Still allow manual parameter goal changes.
        self._check_parameter_goal()

        # Mission is locked: do not publish again. Only wait for board exit.
        if self.mission_locked:
            self._handle_locked_wait_for_exit(image)
            return

        if not self.goal_valid or self.goal_letter is None:
            self.goal_letter = 'A'
            self.goal_valid = True
            self.get_logger().warn(
                "No active goal was set. Falling back to goal 'A'.")

        min_w = self.get_parameter('min_board_width').get_parameter_value().integer_value
        min_h = self.get_parameter('min_board_height').get_parameter_value().integer_value
        conf_threshold = self.get_parameter('confidence_threshold').get_parameter_value().double_value
        required_votes = self.get_parameter('required_consecutive').get_parameter_value().integer_value

        board_reads = read_boards(image)

        if not board_reads:
            # Board gone -> start a fresh vote window for the next board.
            if self.vote_window:
                self.vote_window.clear()
            self.get_logger().info("No mid-range board detected.", throttle_duration_sec=2.0)
            return

        # Multi-board detection is still enabled, but mission decision uses only
        # the forward/center board. This avoids a side board with high confidence
        # changing the mission.
        target_board = select_target_board(board_reads, image.shape)

        best = None
        if target_board is not None:
            x0, y0, x1, y1 = target_board["bbox"]
            board_w = x1 - x0
            board_h = y1 - y0
            if board_w >= min_w and board_h >= min_h:
                direction, confidence = target_board["results"].get(self.goal_letter, (None, 0.0))
                if direction is not None:
                    best = {
                        "direction": direction,
                        "confidence": confidence,
                        "bbox": target_board["bbox"],
                        "board_w": board_w,
                        "board_h": board_h,
                        "boards_seen": len(board_reads),
                    }

        if best is None:
            # No reliable read this frame (quality gate rejected it, or goal
            # cell not found). Do NOT reset the window - just skip voting.
            self.get_logger().info(
                f"Goal '{self.goal_letter}' not read this frame. window={dict(self._window_tally())}",
                throttle_duration_sec=1.0)
            return

        direction = best["direction"]
        confidence = best["confidence"]
        board_w = best["board_w"]
        board_h = best["board_h"]

        # Closeness gate: do NOT vote until the board is close enough to read
        # reliably. Far / mid-range reads are degraded and wrong; voting on them
        # is what causes the oscillation and premature (transitional) locks.
        min_lock_w = self.get_parameter('min_lock_board_width').get_parameter_value().integer_value
        if board_w < min_lock_w:
            self.get_logger().info(
                f"Board {board_w}x{board_h} below min_lock_board_width={min_lock_w}; "
                f"waiting for closer approach (no vote). window={dict(self._window_tally())}",
                throttle_duration_sec=1.0)
            return

        if confidence < conf_threshold:
            self.get_logger().info(
                f"Ignoring {direction}: confidence = {confidence:.2f}, threshold = {conf_threshold:.2f}. window={dict(self._window_tally())}",
                throttle_duration_sec=1.0)
            return

        # ---- VOTE: push into a SLIDING WINDOW of recent reads ----
        # A sliding window (not a running total) so early far-range wrong reads
        # fall off and the close-range correct read can reach a supermajority.
        self.vote_window.append(direction)
        counts = self._window_tally()
        top_dir = max(counts, key=counts.get)
        top_n = counts[top_dir]
        win_n = len(self.vote_window)
        boards_seen = best.get("boards_seen", len(board_reads))

        self.get_logger().info(
            f"Vote {direction}  goal={self.goal_letter}  conf={confidence:.2f}  "
            f"board={board_w}x{board_h}  boards_seen={boards_seen}  "
            f"| window={dict(counts)}  top={top_dir}={top_n}/{win_n}")

        # ---- LOCK: only on a FULL window with a clear supermajority ----
        # Full-window requirement prevents locking on a short early wrong run;
        # the supermajority prevents locking on genuine oscillation.
        if (win_n >= VOTE_WINDOW and
                top_n >= required_votes and
                top_n >= VOTE_SUPERMAJORITY * win_n):

            self.current_mission = top_dir
            self.mission_locked = True
            self.missing_frames = 0

            msg = String()
            msg.data = top_dir
            self.publisher_turn.publish(msg)

            self.get_logger().info(
                f"MISSION LOCKED\n"
                f"Goal: {self.goal_letter}\n"
                f"Direction: {top_dir}\n"
                f"Confidence (last): {confidence:.2f}\n"
                f"Board size: {board_w} x {board_h}\n"
                f"Vote window: {dict(counts)}  (winner {top_n}/{win_n})\n"
                f"State: WAIT_BOARD_EXIT")

    def _window_tally(self):
        counts = {"LEFT": 0, "RIGHT": 0, "STRAIGHT": 0}
        for v in self.vote_window:
            counts[v] = counts.get(v, 0) + 1
        return counts


def main(args=None):
    rclpy.init(args=args)
    node = ObjectRecognizer()
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
