import cv2
import mediapipe as mp
import math
import numpy as np
import time
import threading
import requests
from ultralytics import YOLO

class Tracker:
    def __init__(self, esp32_url: str, send_command_fn=None):
        self.ESP32_URL = esp32_url
        self.send_command_fn = send_command_fn

        self.state_lock = threading.Lock()

        self.active_idx = -1
        self.locked = False

        self.request_toggle_active = False
        self.request_toggle_lock = False

        self.last_gimbal_cmd = None

        self.prev_nose = None
        self.prev_time = None

        self.prev_left_wrist_x = None
        self.prev_right_wrist_x = None

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        print("Loading YOLOv8 model inside Tracker...")
        self.model = YOLO("yolov8n.pt")

        self.prev_r_wrist_z = None
        self.prev_l_wrist_z = None

        self.current_threat_level = 0
        self.current_threat_name = "Idle"

    def calc_distance(self, p1, p2):
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def detect_person(self, frame):
        results = self.model(frame, classes=0)
        person_boxes = []
        person_detected = False

        if results and len(results) > 0:
            boxes = results[0].boxes
            if len(boxes) > 0:
                person_detected = True
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    x = int(x1)
                    y = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)
                    person_boxes.append([x, y, w, h, confidence])

        return person_detected, person_boxes

    def toggle_active(self):
        with self.state_lock:
            self.request_toggle_active = True

    def toggle_lock(self):
        with self.state_lock:
            self.request_toggle_lock = True

    def get_current_threat(self):
        with self.state_lock:
            return {
                "level": self.current_threat_level,
                "name": self.current_threat_name
            }

    def gen_frames(self):
        cap = cv2.VideoCapture(0)# esp32 url "http://192.168.1.90:81/stream"
        if not cap.isOpened():
            raise RuntimeError("Could not open webcam.")

        while True:
            success, frame = cap.read()
            if not success:
                break

            h, w = frame.shape[:2]
            display_img = frame.copy()
            current_time = time.time()

            with self.state_lock:
                if self.request_toggle_active:
                    self.request_toggle_active = False

                if self.request_toggle_lock:
                    if self.active_idx != -1:
                        self.locked = not self.locked
                    self.request_toggle_lock = False

            person_detected, person_boxes = self.detect_person(frame)

            with self.state_lock:
                if self.active_idx >= len(person_boxes):
                    self.active_idx = -1
                    self.locked = False

            for idx, box in enumerate(person_boxes):
                x, y, bw_box, bh_box, confidence = box
                color_box = (0, 255, 0)
                thickness = 2
                with self.state_lock:
                    if idx == self.active_idx:
                        color_box = (0, 165, 255) if not self.locked else (0, 0, 255)
                cv2.rectangle(display_img, (x, y), (x + bw_box, y + bh_box), color_box, thickness)
                cv2.putText(
                    display_img,
                    f"Person: {confidence * 100:.2f}%",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color_box,
                    2
                )

            nose = None
            threat_level = 0
            threat_name = "Idle"
            wave_detected = False
            running_detected = False
            is_right_punch = is_left_punch = is_push = False
            is_right_kick = is_left_kick = False
            is_right_stab = is_left_stab = False
            crowd_detected = False

            if person_detected:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = self.pose.process(image_rgb)
                hand_results = self.hands.process(image_rgb)

                if pose_results.pose_landmarks:
                    lm = pose_results.pose_landmarks.landmark[0]
                    nose = (int(lm.x * w), int(lm.y * h))

                nose_idx = -1
                if nose is not None:
                    nx, ny = nose
                    for idx, (bx, by, bw_box, bh_box, _) in enumerate(person_boxes):
                        if bx <= nx <= bx + bw_box and by <= ny <= by + bh_box:
                            nose_idx = idx
                            break

                if nose_idx != -1 and pose_results.pose_landmarks:
                    landmarks = pose_results.pose_landmarks.landmark

                    def get_point(idx):
                        lm = landmarks[idx]
                        return (int(lm.x * w), int(lm.y * h), lm.z)

                    left_shoulder = get_point(11)
                    right_shoulder = get_point(12)
                    left_elbow = get_point(13)
                    right_elbow = get_point(14)
                    left_wrist = get_point(15)
                    right_wrist = get_point(16)
                    left_hip = get_point(23)
                    right_hip = get_point(24)
                    left_knee = get_point(25)
                    right_knee = get_point(26)
                    left_ankle = get_point(27)
                    right_ankle = get_point(28)

                    if self.prev_nose is not None and self.prev_time is not None:
                        dist = self.calc_distance(nose, self.prev_nose)
                        dt = current_time - self.prev_time
                        if dt > 0:
                            speed = dist / dt
                            if speed > 200:
                                running_detected = True
                    self.prev_nose = nose
                    self.prev_time = current_time

                    if self.prev_left_wrist_x is not None and self.prev_right_wrist_x is not None:
                        if left_wrist[1] < nose[1] and right_wrist[1] < nose[1]:
                            if (abs(left_wrist[0] - self.prev_left_wrist_x) > 20 and
                                abs(right_wrist[0] - self.prev_right_wrist_x) > 20):
                                wave_detected = True
                    self.prev_left_wrist_x = left_wrist[0]
                    self.prev_right_wrist_x = right_wrist[0]

                    def angle_deg(a, b):
                        dot = a[0] * b[0] + a[1] * b[1]
                        mag1 = math.hypot(a[0], a[1]) + 1e-6
                        mag2 = math.hypot(b[0], b[1]) + 1e-6
                        cosang = max(min(dot / (mag1 * mag2), 1.0), -1.0)
                        return math.degrees(math.acos(cosang))

                    down_vec = (0, 1)
                    vec_r = (right_wrist[0] - right_shoulder[0], right_wrist[1] - right_shoulder[1])
                    vec_l = (left_wrist[0] - left_shoulder[0], left_wrist[1] - left_shoulder[1])
                    angle_r = angle_deg(vec_r, down_vec)
                    angle_l = angle_deg(vec_l, down_vec)

                    dz_wr = right_wrist[2] - right_shoulder[2]
                    dz_wl = left_wrist[2] - left_shoulder[2]

                    is_right_punch = (
                        right_wrist[1] < right_elbow[1] - 20
                        and abs(angle_r - 90) < 20
                        and (right_wrist[2] < right_elbow[2] - 0.05)
                    )
                    is_left_punch = (
                        left_wrist[1] < left_elbow[1] - 20
                        and abs(angle_l - 90) < 20
                        and (left_wrist[2] < left_elbow[2] - 0.05)
                    )

                    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
                    wrists_y = (left_wrist[1] + right_wrist[1]) / 2
                    is_push = (
                        abs(wrists_y - shoulder_y) < 40
                        and abs(angle_l - 90) < 25
                        and abs(angle_r - 90) < 25
                        and (left_wrist[2] < left_elbow[2] - 0.03)
                        and (right_wrist[2] < right_elbow[2] - 0.03)
                    )

                    diag_vec_r = (1, 1)
                    diag_vec_l = (-1, 1)
                    if self.prev_r_wrist_z is None:
                        self.prev_r_wrist_z = right_wrist[2]
                        self.prev_l_wrist_z = left_wrist[2]

                    z_speed_r = self.prev_r_wrist_z - right_wrist[2]
                    z_speed_l = self.prev_l_wrist_z - left_wrist[2]
                    self.prev_r_wrist_z = right_wrist[2]
                    self.prev_l_wrist_z = left_wrist[2]

                    z_thrust_thresh = 0.015
                    is_right_stab_std = (
                        right_wrist[1] > right_elbow[1] + 20
                        and angle_deg(vec_r, diag_vec_r) < 30
                        and z_speed_r > z_thrust_thresh
                    )
                    is_left_stab_std = (
                        left_wrist[1] > left_elbow[1] + 20
                        and angle_deg(vec_l, diag_vec_l) < 30
                        and z_speed_l > z_thrust_thresh
                    )
                    is_right_stab_low = (
                        right_wrist[1] > right_elbow[1] + 60
                        and angle_deg(vec_r, diag_vec_r) < 50
                        and z_speed_r > z_thrust_thresh
                    )
                    is_left_stab_low = (
                        left_wrist[1] > left_elbow[1] + 60
                        and angle_deg(vec_l, diag_vec_l) < 50
                        and z_speed_l > z_thrust_thresh
                    )
                    is_right_stab = is_right_stab_std or is_right_stab_low
                    is_left_stab = is_left_stab_std or is_left_stab_low

                    vec_r_leg = (right_ankle[0] - right_hip[0], right_ankle[1] - right_hip[1])
                    vec_l_leg = (left_ankle[0] - left_hip[0], left_ankle[1] - left_hip[1])
                    angle_r_leg = angle_deg(vec_r_leg, down_vec)
                    angle_l_leg = angle_deg(vec_l_leg, down_vec)
                    z_thrust_r_leg = right_knee[2] - right_ankle[2]
                    is_right_kick = (
                        right_ankle[1] < right_knee[1] - 20
                        and abs(angle_r_leg - 90) < 50
                        and z_thrust_r_leg > 0.015
                    )
                    z_thrust_l_leg = left_knee[2] - left_ankle[2]
                    is_left_kick = (
                        left_ankle[1] < left_knee[1] - 20
                        and abs(angle_l_leg - 90) < 50
                        and z_thrust_l_leg > 0.015
                    )

                    is_back_right_arm = (
                        right_wrist[2] > right_shoulder[2] + 0.05
                        and abs(angle_r - 90) < 25
                    )
                    is_back_left_arm = (
                        left_wrist[2] > left_shoulder[2] + 0.05
                        and abs(angle_l - 90) < 25
                    )
                    is_back_right_leg = (
                        right_ankle[2] > right_knee[2] + 0.05
                        and abs(angle_r_leg - 90) < 30
                    )
                    is_back_left_leg = (
                        left_ankle[2] > left_knee[2] + 0.05
                        and abs(angle_l_leg - 90) < 30
                    )

                    if len(person_boxes) > 5:
                        centers = [
                            (bx + bw_box // 2, by + bh_box // 2)
                            for bx, by, bw_box, bh_box, _ in person_boxes
                        ]
                        xs = [c[0] for c in centers]
                        ys = [c[1] for c in centers]
                        if (max(xs) - min(xs) < 200) and (max(ys) - min(ys) < 200):
                            crowd_detected = True

                    if is_right_stab or is_left_stab or wave_detected:
                        threat_level = 4
                        threat_name = "Stab" if not wave_detected else "Wave"
                    elif any([is_right_punch, is_left_punch, is_push,
                              is_right_kick, is_left_kick,
                              is_back_right_arm, is_back_left_arm,
                              is_back_right_leg, is_back_left_leg]):
                        if is_right_punch or is_left_punch:
                            threat_name = "Punch"
                        elif is_push:
                            threat_name = "Push"
                        else:
                            threat_name = "Kick"
                        threat_level = 3
                    elif crowd_detected:
                        threat_level = 2
                        threat_name = "Crowd"
                    elif running_detected:
                        threat_level = 1
                        threat_name = "Running"
                    else:
                        threat_level = 0
                        threat_name = "Idle"

                    with self.state_lock:
                        self.current_threat_level = threat_level
                        self.current_threat_name = threat_name

                    with self.state_lock:
                        if not self.locked and nose_idx != -1:
                            if threat_level in [1, 2, 3]:
                                self.active_idx = nose_idx
                            elif threat_level == 4:
                                self.active_idx = nose_idx
                                self.locked = True

                    if threat_name == "Wave":
                        color = (0, 0, 255)
                    elif threat_level == 4:
                        color = (0, 0, 255)
                    elif threat_level == 3:
                        color = (0, 128, 255)
                    elif threat_level == 2:
                        color = (0, 165, 255)
                    elif threat_level == 1:
                        color = (0, 255, 255)
                    else:
                        color = (0, 255, 0)

                    cv2.putText(
                        display_img,
                        f"{'Wave' if threat_name == 'Wave' else f'Level {threat_level}: {threat_name}'}",
                        (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        3
                    )

                    with self.state_lock:
                        if self.locked and self.active_idx != -1:
                            bx, by, bw_box, bh_box, _ = person_boxes[self.active_idx]
                            person_center_y = (by + bh_box // 2) - 60
                            frame_center_y = h // 2
                            tol = 20
                            cmd = None
                            if person_center_y < frame_center_y - tol:
                                cmd = "U"
                            elif person_center_y > frame_center_y + tol:
                                cmd = "D"

                            if cmd and cmd != self.last_gimbal_cmd:
                                self.last_gimbal_cmd = cmd
                                try:
                                    if self.send_command_fn:
                                        self.send_command_fn(cmd)
                                    else:
                                        requests.get(f"{self.ESP32_URL}?cmd={cmd}", timeout=0.5)
                                except requests.RequestException:
                                    pass

                            if cmd:
                                cv2.putText(
                                    display_img,
                                    f"CMD={cmd}",
                                    (30, 160),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1,
                                    (0,0,0),
                                    3
                                )

                with self.state_lock:
                    if self.request_toggle_active:
                        if person_detected and len(person_boxes) > 0:
                            self.active_idx = (self.active_idx + 1) % len(person_boxes)
                            self.locked = False
                        self.request_toggle_active = False

                for idx, box in enumerate(person_boxes):
                    x, y, bw_box, bh_box, confidence = box
                    color_box = (0, 255, 0)
                    thickness = 2
                    with self.state_lock:
                        if idx == self.active_idx:
                            color_box = (0, 165, 255) if not self.locked else (0, 0, 255)
                    cv2.rectangle(display_img, (x, y), (x + bw_box, y + bh_box), color_box, thickness)
                    cv2.putText(
                        display_img,
                        f"Person: {confidence * 100:.2f}%",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color_box,
                        2
                    )

            else:
                cv2.putText(
                    display_img,
                    "No person detected",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

            ret, buffer = cv2.imencode('.jpg', display_img)
            frame_bytes = buffer.tobytes()

            yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' +
                frame_bytes +
                b'\r\n'
            )

        cap.release()