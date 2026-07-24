# Copyright 2024-2026 NXP
# Copyright 2016 Open Source Robotics Foundation, Inc.
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
import time
import math
from sensor_msgs.msg import Joy, LaserScan
from std_msgs.msg import String
from synapse_msgs.msg import EdgeVectors, ServerCommunication

QOS_PROFILE_DEFAULT = 10
PI = math.pi

# Control bounds
SPEED_MIN = 0.0
SPEED_MAX = 1.0
TURN_MIN = -1.0
TURN_MAX = 1.0

# CONFIGURATION:
# The buggy is driven in manual mode by publishing standard controller Joy messages to /cerebri/in/joy.
# The layout is: msg.axes = [0.0, speed, 0.0, turn]
# - speed: positive for forward, negative for reverse. Range: [-1.0, 1.0]
# - turn: positive for left steer, negative for right steer. Range: [-1.0, 1.0]
# msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1] (Keep buttons set to this pattern for manual override mode)

class LineFollower(Node):
    """
    Core controller Node for the B3RB buggy.
    By default, it publishes a safe drive-straight command on a timer loop.
    Implement logic inside the callbacks to steer, dodge obstacles, detect destinations,
    communicate with the server, and park.
    """
    def __init__(self):
        super().__init__('line_follower')

        # ------------------ Subscriptions ------------------

        # 1. Lane Edge Vectors (from edge_vectors_publisher)
        self.subscription_vectors = self.create_subscription(
            EdgeVectors,
            '/edge_vectors',
            self.edge_vectors_callback,
            QOS_PROFILE_DEFAULT)

        # 2. LIDAR Obstacle Scanner
        self.subscription_lidar = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            QOS_PROFILE_DEFAULT)

        # 3. Server Communication Feedback Loop
        self.subscription_server = self.create_subscription(
            ServerCommunication,
            '/ServerCommunication',
            self.server_communication_callback,
            QOS_PROFILE_DEFAULT)

        # 4. QR Code Detections (from qr_detector)
        self.subscription_qr = self.create_subscription(
            String,
            '/qr_detection',
            self.qr_detection_callback,
            QOS_PROFILE_DEFAULT)

        # 5. Sign Board Detections (from object_recognizer)
        self.subscription_signs = self.create_subscription(
            String,
            '/sign_board_detection',
            self.sign_board_callback,
            QOS_PROFILE_DEFAULT)

        # ------------------ Publishers ------------------

        # Publisher to drive/steer the buggy
        self.publisher_joy = self.create_publisher(
            Joy,
            '/cerebri/in/joy',
            QOS_PROFILE_DEFAULT)

        # Publisher to send messages to the Server
        self.publisher_server = self.create_publisher(
            ServerCommunication,
            '/ServerCommunication',
            QOS_PROFILE_DEFAULT)

        # ------------------ State Variables & Timer ------------------

        # Default controls: drive straight slowly
        self.target_speed = 0.15
        self.target_turn = 0.0

        # State variables (You can add your own state flags / state machines here)
        self.obstacle_in_front = False
        self.patient_id = None
        self.hospital_id = None
        self.current_destination = None
        self.mission_completed = False

        # ------------------ Task 1: PID Lane Following Parameters ------------------

        # PID gains exposed as ROS parameters.
        # NOTE: gains now act on a NORMALIZED error in [-1, 1], not raw pixel error.
        self.declare_parameter('Kp', 1.0)
        self.declare_parameter('Ki', 0.0)
        self.declare_parameter('Kd', 0.35)

        # Lane / steering / speed tuning parameters.
        self.declare_parameter('estimated_lane_width_px', 120.0)
        self.declare_parameter('integral_clamp', 1.0)
        self.declare_parameter('steering_smoothing_alpha', 0.4)
        self.declare_parameter('speed_smoothing_alpha', 0.3)

        self.declare_parameter('speed_straight', 0.45)
        self.declare_parameter('speed_medium', 0.35)
        self.declare_parameter('speed_sharp', 0.25)
        self.declare_parameter('steering_threshold_medium', 0.25)
        self.declare_parameter('steering_threshold_sharp', 0.55)

        self.declare_parameter('no_vector_hold_timeout', 0.5)
        self.declare_parameter('no_vector_decay_rate', 0.85)
        self.declare_parameter('no_vector_speed', 0.15)

        # Steering direction can flip depending on hardware/camera mounting.
        self.declare_parameter('invert_steering', False)

        # Debug log throttling.
        self.declare_parameter('debug_log_period', 1.0)

        # Cache parameter values for fast access in callbacks.
        self.Kp = self.get_parameter('Kp').value
        self.Ki = self.get_parameter('Ki').value
        self.Kd = self.get_parameter('Kd').value

        self.estimated_lane_width_px = self.get_parameter('estimated_lane_width_px').value
        self.integral_clamp = self.get_parameter('integral_clamp').value
        self.steering_smoothing_alpha = self.get_parameter('steering_smoothing_alpha').value
        self.speed_smoothing_alpha = self.get_parameter('speed_smoothing_alpha').value

        self.speed_straight = self.get_parameter('speed_straight').value
        self.speed_medium = self.get_parameter('speed_medium').value
        self.speed_sharp = self.get_parameter('speed_sharp').value
        self.steering_threshold_medium = self.get_parameter('steering_threshold_medium').value
        self.steering_threshold_sharp = self.get_parameter('steering_threshold_sharp').value

        self.no_vector_hold_timeout = self.get_parameter('no_vector_hold_timeout').value
        self.no_vector_decay_rate = self.get_parameter('no_vector_decay_rate').value
        self.no_vector_speed = self.get_parameter('no_vector_speed').value

        self.invert_steering = self.get_parameter('invert_steering').value
        self.debug_log_period = self.get_parameter('debug_log_period').value

        # PID controller persistent state.
        self.previous_error = 0.0
        self.integral = 0.0
        self.previous_time = None

        # Output smoothing (exponential filtering) state.
        self.filtered_steering = 0.0
        self.filtered_speed = 0.0

        # No-vector hold/decay state.
        self.last_valid_steering = 0.0
        self.last_vector_time = None

        # Debug log throttling state.
        self.last_debug_log_time = 0.0

        # Timer to publish drive commands at 10Hz
        self.control_timer = self.create_timer(0.1, self.publish_drive_commands)

        self.get_logger().info("Line Follower controller initialized. Safe Drive-Straight Mode active.")

    def publish_drive_commands(self):
        """Timer callback that periodically publishes the current speed and steer command."""
        msg = Joy()
        msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]  # Manual override button configuration
        msg.axes = [0.0, self.target_speed, 0.0, self.target_turn]
        self.publisher_joy.publish(msg)

    def rover_move_manual_mode(self, speed, turn):
        """Helper to immediately set control speed and steering angle."""
        self.target_speed = float(max(min(speed, SPEED_MAX), -SPEED_MAX))
        self.target_turn = float(max(min(turn, TURN_MAX), -TURN_MAX))

    # ------------------ Task 1: PID Lane Following Helpers ------------------

    @staticmethod
    def _bottom_point(vector):
        """
        Given an edge vector (two endpoints), return the endpoint with
        the LARGER y value, i.e. the point closest to the buggy.
        Does NOT assume a fixed index ordering.
        """
        p0, p1 = vector[0], vector[1]
        return p1 if p1.y >= p0.y else p0

    def compute_error(self, message):
        """
        Compute the lane-center steering error from an EdgeVectors message.

        Does NOT assume vector_1 == left / vector_2 == right. When two
        vectors are present, they are sorted by x-coordinate of their
        bottom point so the smaller x is treated as the left boundary
        and the larger x as the right boundary.

        - vector_count == 2: lane_center is the midpoint of the two
          bottom points (after sorting by x).
        - vector_count == 1: lane_center is estimated by offsetting the
          single bottom point by half the configured lane width,
          depending on whether it lies left or right of image center.
        - vector_count == 0: no error can be computed.

        Args:
            message (EdgeVectors): Lane edge vectors message.

        Returns:
            tuple(float, float, float, float, bool): (lane_center,
            left_x, right_x, image_center, vectors_available).
            left_x/right_x are None when not applicable.
        """
        image_width = float(message.image_width)
        image_center = image_width / 2.0
        vector_count = message.vector_count

        if vector_count >= 2:
            x1 = self._bottom_point(message.vector_1).x
            x2 = self._bottom_point(message.vector_2).x

            # Sort by x-coordinate: smaller x -> left boundary,
            # larger x -> right boundary. Independent of message ordering.
            left_x, right_x = (x1, x2) if x1 <= x2 else (x2, x1)

            lane_center = (left_x + right_x) / 2.0
            return lane_center, left_x, right_x, image_center, True

        elif vector_count == 1:
            single_x = self._bottom_point(message.vector_1).x
            half_lane_width = self.estimated_lane_width_px / 2.0

            if single_x < image_center:
                # Single detected vector is the left edge; estimate
                # center by shifting right by half the lane width.
                lane_center = single_x + half_lane_width
                left_x, right_x = single_x, None
            else:
                # Single detected vector is the right edge; estimate
                # center by shifting left by half the lane width.
                lane_center = single_x - half_lane_width
                left_x, right_x = None, single_x

            return lane_center, left_x, right_x, image_center, True

        else:
            return image_center, None, None, image_center, False

    def compute_pid(self, normalized_error, now, vectors_available):
        """
        Run the PID controller (with hold/decay fallback when no
        vectors are available) to produce a raw, clamped steering
        command. Operates on a NORMALIZED error in [-1, 1] so gains
        are independent of camera resolution.

        Maintains self.previous_error, self.integral, and
        self.previous_time as persistent controller state. The
        integral term is clamped for anti-windup, and dt is bounded
        to avoid derivative spikes from irregular callback timing.

        Args:
            normalized_error (float): Lane-center error normalized to [-1, 1].
            now (float): Current time in seconds (time.time()).
            vectors_available (bool): Whether valid vectors were seen.

        Returns:
            float: Raw steering command, clamped to [-1.0, 1.0].
        """
        if not vectors_available:
            return self._handle_no_vectors(now)

        if self.previous_time is None:
            dt = 0.033  # assume ~30Hz on first sample
        else:
            dt = now - self.previous_time
            # Guard against non-positive or excessively large dt
            # (clock jumps, startup jitter) to prevent derivative spikes.
            if dt <= 0.0 or dt > 0.5:
                dt = 0.033

        # Proportional term.
        p_term = self.Kp * normalized_error

        # Integral term with anti-windup clamping.
        self.integral += normalized_error * dt
        self.integral = max(-self.integral_clamp, min(self.integral_clamp, self.integral))
        i_term = self.Ki * self.integral

        # Derivative term.
        derivative = (normalized_error - self.previous_error) / dt
        d_term = self.Kd * derivative

        steering = p_term + i_term + d_term

        if self.invert_steering:
            steering = -steering

        # Update persistent controller state.
        self.previous_error = normalized_error
        self.previous_time = now
        self.last_valid_steering = steering

        # Clamp final steering command.
        steering = max(TURN_MIN, min(TURN_MAX, steering))

        return steering

    def _handle_no_vectors(self, now):
        """
        Handle the case where no edge vectors are available.

        Holds the last known steering command for a short configurable
        timeout, then exponentially decays the steering toward zero.
        Resets PID timing/integral state so the controller does not
        spike when vectors reappear.

        Args:
            now (float): Current time in seconds (time.time()).

        Returns:
            float: Steering command to use while no vectors are seen.
        """
        # Reset timing so dt is recomputed cleanly once vectors return.
        self.previous_time = None
        self.integral = 0.0
        self.previous_error = 0.0

        if self.last_vector_time is None:
            elapsed_since_last = float('inf')
        else:
            elapsed_since_last = now - self.last_vector_time

        if elapsed_since_last <= self.no_vector_hold_timeout:
            # Hold the last known steering briefly.
            steering = self.last_valid_steering
        else:
            # Slowly decay steering toward zero.
            self.last_valid_steering *= self.no_vector_decay_rate
            if abs(self.last_valid_steering) < 1e-3:
                self.last_valid_steering = 0.0
            steering = self.last_valid_steering

        steering = max(TURN_MIN, min(TURN_MAX, steering))
        return steering

    def compute_speed(self, steering, vectors_available, now):
        """
        Compute an adaptive target speed based on steering magnitude.

        Small steering (near-straight lane) -> higher speed.
        Large steering (sharp curve) -> lower speed.
        When no vectors have been seen beyond the hold timeout, speed
        is reduced toward a low crawl speed (never stopped abruptly).

        Args:
            steering (float): Current (raw) steering command.
            vectors_available (bool): Whether valid vectors were seen.
            now (float): Current time in seconds (time.time()).

        Returns:
            float: Target speed in [0.0, speed_straight].
        """
        if not vectors_available:
            elapsed_since_last = (
                float('inf') if self.last_vector_time is None
                else now - self.last_vector_time
            )
            if elapsed_since_last > self.no_vector_hold_timeout:
                return self.no_vector_speed

        abs_steering = abs(steering)

        if abs_steering >= self.steering_threshold_sharp:
            speed = self.speed_sharp
        elif abs_steering >= self.steering_threshold_medium:
            speed = self.speed_medium
        else:
            speed = self.speed_straight

        return speed

    # ------------------ Callback Implementations ------------------

    def edge_vectors_callback(self, message):
        """
        Receives lane boundaries from the camera vector extractor and
        performs Task 1 (camera-based PID lane following).

        Computes the lane-center error (order-independent, bottom-point
        based), normalizes it to [-1, 1], runs it through the PID
        controller (with hold/decay fallback when no vectors are
        seen), derives an adaptive speed from the steering magnitude,
        applies exponential smoothing to both, and commits the result
        via rover_move_manual_mode().
        """
        now = time.time()

        lane_center, left_x, right_x, image_center, vectors_available = self.compute_error(message)

        if image_center > 0.0:
            normalized_error = (lane_center - image_center) / image_center
        else:
            normalized_error = 0.0
        normalized_error = max(-1.0, min(1.0, normalized_error))

        if vectors_available:
            self.last_vector_time = now

        raw_steering = self.compute_pid(normalized_error, now, vectors_available)
        raw_speed = self.compute_speed(raw_steering, vectors_available, now)

        # Exponential smoothing: filtered = alpha * new + (1 - alpha) * filtered.
        alpha_s = self.steering_smoothing_alpha
        alpha_v = self.speed_smoothing_alpha

        self.filtered_steering = alpha_s * raw_steering + (1.0 - alpha_s) * self.filtered_steering
        self.filtered_speed = alpha_v * raw_speed + (1.0 - alpha_v) * self.filtered_speed

        # Final clamp for safety before committing.
        self.filtered_steering = max(TURN_MIN, min(TURN_MAX, self.filtered_steering))
        self.filtered_speed = max(SPEED_MIN, min(SPEED_MAX, self.filtered_speed))

        self.rover_move_manual_mode(self.filtered_speed, self.filtered_steering)

        # Throttled debug logging (once per debug_log_period seconds) for PID tuning.
        if now - self.last_debug_log_time >= self.debug_log_period:
            self.last_debug_log_time = now
            self.get_logger().info(
                f"vector_count={message.vector_count} "
                f"left_x={left_x} right_x={right_x} "
                f"lane_center={lane_center:.2f} image_center={image_center:.2f} "
                f"normalized_error={normalized_error:.4f} "
                f"steering={self.filtered_steering:.4f} speed={self.filtered_speed:.4f}"
            )

    def lidar_callback(self, message):
        """
        Placeholder for Task 2 (Obstacle Avoidance & Building Range).
        Not implemented per current task scope (lane following only).
        """
        # HINTS:
        # num_readings = len(message.ranges)
        # front_sector = message.ranges[int(num_readings * 7/18): int(num_readings * 11/18)]
        # min_front_dist = min(front_sector)
        pass

    def server_communication_callback(self, message):
        """
        Placeholder for Task 3 (Server Communication).
        Not implemented per current task scope (lane following only).
        """
        if message.dest == 1:
            self.get_logger().info(f"Received Server Message: {message.msg}")
            # Parse payload and update state machine destination/objectives here
            pass

    def send_server_update(self, text_msg):
        """Sends status messages to the server. (Do not forget to send ACK messages to server)"""
        server_msg = ServerCommunication()
        server_msg.src = 1       # Source component: Buggy-1
        server_msg.dest = 2      # Destination component: Server-2
        server_msg.uid = 100     # Replace with a rolling message ID/counter
        server_msg.ack = 0
        server_msg.msg = text_msg
        self.publisher_server.publish(server_msg)

    def qr_detection_callback(self, message):
        """
        Placeholder for Task 4 (Patient/Hospital Identification via QR).
        Not implemented per current task scope (lane following only).
        """
        self.get_logger().info(f"Heard QR code: {message.data}")
        pass

    def sign_board_callback(self, message):
        """
        Placeholder for Task 5 (Sign Board Routing).
        Not implemented per current task scope (lane following only).
        """
        self.get_logger().info(f"Heard Sign Board: {message.data}")
        pass

def main(args=None):
    rclpy.init(args=args)
    node = LineFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
