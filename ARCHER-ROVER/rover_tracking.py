import cv2
import numpy as np
import serial
import time
import threading
import os
import requests
from collections import deque


class AdaptiveMultiTracker:
    def __init__(self):
        self.primary_tracker = None
        self.backup_tracker = None
        self.tracker_type = 'CSRT'
        self.backup_type = 'KCF'

        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)

        self.position_history = deque(maxlen=10)
        self.velocity_history = deque(maxlen=5)
        self.confidence_history = deque(maxlen=8)
        self.size_history = deque(maxlen=5)

        self.scale_factors = [1.0, 0.8, 1.2]

        self.search_window_scale = 1.5
        self.min_search_scale = 1.2
        self.max_search_scale = 3.0

        self.tracking_confidence = 0.0
        self.detection_confidence = 0.0
        self.motion_confidence = 0.0
        self.overall_confidence = 0.0

        self.tracking_state = 'SEARCHING'
        self.frames_since_detection = 0
        self.prediction_frames = 0
        self.max_prediction_frames = 25
        self.consecutive_low_confidence = 0

        self.prediction_confidence_threshold = 0.05
        self.tracking_loss_threshold = 0.02
        self.max_frames_without_detection = 60
        self.min_consecutive_low_conf = 15

        self.roi_expansion_rate = 1.2
        self.roi_contraction_rate = 0.95

        self.is_moving_fast = False
        self.movement_threshold = 15

    def initialize_tracking(self, frame, bbox, detection_confidence=1.0):
        x, y, w, h = bbox

        self.primary_tracker = self._create_tracker(self.tracker_type)
        self.primary_tracker.init(frame, bbox)

        self.backup_tracker = self._create_tracker(self.backup_type)
        self.backup_tracker.init(frame, bbox)

        cx, cy = x + w / 2, y + h / 2
        self.kalman.statePre = np.array([cx, cy, 0, 0], np.float32)
        self.kalman.statePost = np.array([cx, cy, 0, 0], np.float32)

        self.position_history.clear()
        self.velocity_history.clear()
        self.confidence_history.clear()
        self.size_history.clear()

        self.position_history.append((cx, cy, w, h))
        self.size_history.append((w, h))
        self.confidence_history.append(detection_confidence)

        self.tracking_state = 'TRACKING'
        self.frames_since_detection = 0
        self.consecutive_low_confidence = 0
        self.detection_confidence = detection_confidence
        self.is_moving_fast = False

        print(f"[TRACKER] Initialized tracking at ({x}, {y}) size ({w}x{h})")
        return True

    def _create_tracker(self, tracker_type):
        if tracker_type == 'CSRT':
            return cv2.TrackerCSRT_create()
        elif tracker_type == 'KCF':
            return cv2.TrackerKCF_create()
        elif tracker_type == 'MOSSE':
            return cv2.TrackerMOSSE_create()
        else:
            return cv2.TrackerCSRT_create()

    def update(self, frame, detections=None):
        if self.tracking_state == 'SEARCHING':
            return self._handle_searching_state(frame, detections)
        elif self.tracking_state == 'TRACKING':
            return self._handle_tracking_state(frame, detections)
        elif self.tracking_state == 'PREDICTING':
            return self._handle_predicting_state(frame, detections)
        else:
            return self._handle_lost_state(frame, detections)

    def _handle_tracking_state(self, frame, detections):
        success_primary, bbox_primary = self.primary_tracker.update(frame)
        success_backup, bbox_backup = self.backup_tracker.update(frame)

        primary_conf = self._calculate_tracking_confidence(frame, bbox_primary) if success_primary else 0.0
        backup_conf = self._calculate_tracking_confidence(frame, bbox_backup) if success_backup else 0.0

        print(
            f"[TRACKER] Primary: {success_primary} ({primary_conf:.3f}), Backup: {success_backup} ({backup_conf:.3f})")

        chosen_bbox = None
        chosen_conf = 0.0

        if success_primary and primary_conf > 0.05:
            chosen_bbox = bbox_primary
            chosen_conf = primary_conf
            self.tracking_confidence = primary_conf
        elif success_backup and backup_conf > 0.03:
            chosen_bbox = bbox_backup
            chosen_conf = backup_conf
            self.tracking_confidence = backup_conf

        if chosen_bbox is not None:
            self._detect_movement(chosen_bbox)

            self._update_motion_model(chosen_bbox)

            best_detection = self._find_matching_detection(chosen_bbox, detections)

            if best_detection is not None:
                det_bbox, det_conf = best_detection
                self._reinforce_tracking(frame, det_bbox, det_conf)
                self.frames_since_detection = 0
                self.detection_confidence = det_conf
                self.consecutive_low_confidence = 0
                print(f"[TRACKER] Reinforced with detection (conf: {det_conf:.3f})")
            else:
                self.frames_since_detection += 1
                if chosen_conf < 0.03 and not self.is_moving_fast:
                    self.consecutive_low_confidence += 1
                else:
                    self.consecutive_low_confidence = max(0, self.consecutive_low_confidence - 2)

            self._update_overall_confidence()

            should_predict = (
                    (self.overall_confidence < self.prediction_confidence_threshold and
                     self.consecutive_low_confidence >= self.min_consecutive_low_conf and
                     not self.is_moving_fast) or
                    (self.frames_since_detection > self.max_frames_without_detection and
                     not self.is_moving_fast)
            )

            if should_predict:
                print(f"[TRACKER] Switching to PREDICTION mode. Overall conf: {self.overall_confidence:.3f}, "
                      f"Low conf frames: {self.consecutive_low_confidence}, "
                      f"Frames since detection: {self.frames_since_detection}")
                self.tracking_state = 'PREDICTING'
                self.prediction_frames = 0
                return self._handle_predicting_state(frame, detections)

            return True, chosen_bbox, max(0.3, self.overall_confidence)

        else:
            print("[TRACKER] Both trackers failed completely, switching to PREDICTION")
            self.tracking_state = 'PREDICTING'
            self.prediction_frames = 0
            return self._handle_predicting_state(frame, detections)

    def _detect_movement(self, bbox):
        if len(self.position_history) < 2:
            self.is_moving_fast = False
            return

        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2

        position_list = list(self.position_history)
        if len(position_list) >= 1:
            prev_pos = position_list[-1]
            movement = np.hypot(cx - prev_pos[0], cy - prev_pos[1])
            self.is_moving_fast = movement > self.movement_threshold

            if self.is_moving_fast:
                print(f"[TRACKER] Fast movement detected: {movement:.1f} pixels")

    def _calculate_tracking_confidence(self, frame, bbox, template_size=(50, 50)):
        if bbox is None:
            return 0.0

        x, y, w, h = map(int, bbox)

        if x < -50 or y < -50 or x + w >= frame.shape[1] + 50 or y + h >= frame.shape[0] + 50:
            return 0.0

        x = max(0, min(x, frame.shape[1] - 1))
        y = max(0, min(y, frame.shape[0] - 1))
        w = max(1, min(w, frame.shape[1] - x))
        h = max(1, min(h, frame.shape[0] - y))

        if w <= 0 or h <= 0:
            return 0.1

        try:
            current_patch = frame[y:y + h, x:x + w]
            if current_patch.size == 0:
                return 0.1

            current_patch_resized = cv2.resize(current_patch, template_size)
        except:
            return 0.1

        try:
            gray_patch = cv2.cvtColor(current_patch_resized, cv2.COLOR_BGR2GRAY) if len(
                current_patch_resized.shape) == 3 else current_patch_resized
            variance = np.var(gray_patch)
            texture_confidence = min(1.0, max(0.2, variance / 800.0))
        except:
            texture_confidence = 0.3

        size_confidence = 0.8
        if len(self.size_history) > 3:
            recent_sizes = list(self.size_history)[-2:]
            if len(recent_sizes) > 0:
                avg_area = np.mean([s[0] * s[1] for s in recent_sizes])
                current_area = w * h
                if avg_area > 0:
                    size_ratio = min(current_area, avg_area) / max(current_area, avg_area)
                    size_confidence = max(0.6, size_ratio)

        motion_confidence = self._calculate_motion_consistency(bbox)

        boundary_penalty = 1.0
        edge_margin = 10
        if (x < edge_margin or y < edge_margin or
                x + w > frame.shape[1] - edge_margin or
                y + h > frame.shape[0] - edge_margin):
            boundary_penalty = 0.95

        if self.is_moving_fast:
            overall = (texture_confidence * 0.8 +
                       size_confidence * 0.1 +
                       motion_confidence * 0.05 +
                       boundary_penalty * 0.05)
        else:
            overall = (texture_confidence * 0.6 +
                       size_confidence * 0.2 +
                       motion_confidence * 0.1 +
                       boundary_penalty * 0.1)

        return np.clip(overall, 0.1, 1.0)

    def _calculate_motion_consistency(self, bbox):
        if len(self.position_history) < 2:
            return 0.9

        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2

        position_list = list(self.position_history)

        if len(position_list) >= 1:
            prev_pos = position_list[-1]
            current_velocity = np.hypot(cx - prev_pos[0], cy - prev_pos[1])

            if current_velocity > 10:
                return 0.9
            elif current_velocity > 5:
                return 0.8
            else:
                return 0.7

        return 0.8

    def _update_motion_model(self, bbox):
        x, y, w, h = bbox
        cx, cy = x + w / 2, y + h / 2

        measurement = np.array([[cx], [cy]], np.float32)
        self.kalman.correct(measurement)

        self.position_history.append((cx, cy, w, h))
        self.size_history.append((w, h))

        if len(self.position_history) >= 2:
            position_list = list(self.position_history)
            prev_pos = position_list[-2]
            current_pos = position_list[-1]
            vx = current_pos[0] - prev_pos[0]
            vy = current_pos[1] - prev_pos[1]
            self.velocity_history.append((vx, vy))

            if len(self.velocity_history) > 0:
                recent_vels = list(self.velocity_history)[-3:]
                avg_vx = np.mean([v[0] for v in recent_vels])
                avg_vy = np.mean([v[1] for v in recent_vels])

                self.kalman.statePost[2] = avg_vx
                self.kalman.statePost[3] = avg_vy

    def _find_matching_detection(self, tracking_bbox, detections, expand_search=False):
        if detections is None or len(detections) == 0:
            return None

        tx, ty, tw, th = tracking_bbox
        tcx, tcy = tx + tw / 2, ty + th / 2

        base_radius = 200 if not expand_search else 350
        if self.is_moving_fast:
            base_radius *= 2

        best_match = None
        best_score = 0.0

        for detection in detections:
            if len(detection) == 2:
                det_bbox, det_conf = detection
            else:
                continue

            dx, dy, dw, dh = det_bbox
            dcx, dcy = dx + dw / 2, dy + dh / 2

            distance = np.hypot(tcx - dcx, tcy - dcy)
            if distance > base_radius:
                continue

            iou = self._calculate_iou(tracking_bbox, det_bbox)

            size_ratio = min(tw * th, dw * dh) / max(tw * th, dw * dh)
            if size_ratio < 0.15:
                continue

            distance_factor = max(0.05, 1.0 - distance / base_radius)
            score = (det_conf * 0.4 + iou * 0.2 + size_ratio * 0.2 + distance_factor * 0.2)

            if score > best_score and score > 0.1:
                best_score = score
                best_match = (det_bbox, det_conf)

        return best_match

    def _calculate_iou(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        xi = max(x1, x2)
        yi = max(y1, y2)
        wi = max(0, min(x1 + w1, x2 + w2) - xi)
        hi = max(0, min(y1 + h1, y2 + h2) - yi)
        intersection = wi * hi

        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection

        return intersection / (union + 1e-6)

    def _reinforce_tracking(self, frame, det_bbox, det_conf):
        self.primary_tracker = self._create_tracker(self.tracker_type)
        self.primary_tracker.init(frame, det_bbox)

        self.backup_tracker = self._create_tracker(self.backup_type)
        self.backup_tracker.init(frame, det_bbox)

        x, y, w, h = det_bbox
        self.size_history.append((w, h))

        self.search_window_scale = self.min_search_scale

    def _update_overall_confidence(self):
        if len(self.confidence_history) > 0:
            conf_list = list(self.confidence_history)
            weights = np.linspace(0.7, 1.0, len(conf_list))
            weighted_conf = np.average(conf_list, weights=weights)
        else:
            weighted_conf = 0.5

        detection_weight = min(1.0, max(0.6, 1.0 - self.frames_since_detection * 0.01))

        self.overall_confidence = (
                self.tracking_confidence * 0.7 +
                self.detection_confidence * detection_weight * 0.2 +
                weighted_conf * 0.1
        )

        if self.is_moving_fast:
            self.overall_confidence = max(0.5, self.overall_confidence * 1.2)

        if self.tracking_confidence > 0.1:
            self.overall_confidence = max(0.4, self.overall_confidence)

        self.confidence_history.append(self.overall_confidence)

    def _handle_predicting_state(self, frame, detections):
        self.prediction_frames += 1
        print(f"[TRACKER] PREDICTING frame {self.prediction_frames}/{self.max_prediction_frames}")

        prediction = self.kalman.predict()
        pred_x, pred_y = int(prediction[0]), int(prediction[1])

        if len(self.size_history) > 0:
            recent_sizes = list(self.size_history)
            avg_w = int(np.mean([s[0] for s in recent_sizes]))
            avg_h = int(np.mean([s[1] for s in recent_sizes]))
        else:
            avg_w, avg_h = 100, 100

        predicted_bbox = (pred_x - avg_w // 2, pred_y - avg_h // 2, avg_w, avg_h)

        best_detection = self._find_matching_detection(predicted_bbox, detections, expand_search=True)

        if best_detection is not None:
            det_bbox, det_conf = best_detection
            print(f"[TRACKER] Reacquired target in prediction mode (conf: {det_conf:.3f})")
            self.initialize_tracking(frame, det_bbox, det_conf)
            return True, det_bbox, det_conf

        if self.prediction_frames > self.max_prediction_frames:
            print("[TRACKER] Prediction timeout, switching to LOST")
            self.tracking_state = 'LOST'
            return False, None, 0.0

        self.overall_confidence = max(0.2, 0.8 - (self.prediction_frames * 0.02))

        measurement = np.array([[pred_x], [pred_y]], np.float32)
        self.kalman.correct(measurement)

        return True, predicted_bbox, self.overall_confidence

    def _handle_searching_state(self, frame, detections):
        if detections is None or len(detections) == 0:
            return False, None, 0.0

        best_detection = max(detections, key=lambda x: x[1])

        if best_detection[1] > 0.4:
            self.initialize_tracking(frame, best_detection[0], best_detection[1])
            return True, best_detection[0], best_detection[1]

        return False, None, 0.0

    def _handle_lost_state(self, frame, detections):
        if detections is not None and len(detections) > 0:
            best_detection = max(detections, key=lambda x: x[1])
            if best_detection[1] > 0.5:
                print(f"[TRACKER] Reacquired target from LOST state (conf: {best_detection[1]:.3f})")
                self.initialize_tracking(frame, best_detection[0], best_detection[1])
                return True, best_detection[0], best_detection[1]

        return False, None, 0.0

    def get_state_info(self):
        return {
            'state': self.tracking_state,
            'overall_confidence': self.overall_confidence,
            'tracking_confidence': self.tracking_confidence,
            'detection_confidence': self.detection_confidence,
            'frames_since_detection': self.frames_since_detection,
            'consecutive_low_confidence': self.consecutive_low_confidence,
            'prediction_frames': self.prediction_frames if self.tracking_state == 'PREDICTING' else 0,
            'is_moving_fast': self.is_moving_fast
        }


class RoverTrackingSystem:
    def __init__(self, camera_index=0, serial_port='/dev/ttyUSB0', baud_rate=115200):
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise Exception("Could not open video device")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        try:
            self.ser = serial.Serial(serial_port, baud_rate, timeout=1)
            print(f"Connected to Arduino on {serial_port}")
        except Exception as e:
            print(f"Failed to connect to serial port: {e}")
            self.ser = None

        self.model_path = os.path.dirname(os.path.abspath(__file__))
        self.prototxt_path = os.path.join(self.model_path, 'MobileNetSSD_deploy.prototxt')
        self.model_weights_path = os.path.join(self.model_path, 'MobileNetSSD_deploy.caffemodel')
        if not (os.path.exists(self.prototxt_path) and os.path.exists(self.model_weights_path)):
            raise Exception("MobileNet SSD model files are missing in project directory")

        self.net = cv2.dnn.readNetFromCaffe(self.prototxt_path, self.model_weights_path)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.conf_threshold = 0.38
        self.detected_people = []
        self.active_person_index = -1
        self.locked_person_index = -1
        self.tracking = False
        self.tracker = cv2.TrackerCSRT_create()
        self.tracking_initialized = False

        self.gimbal_angle = 90
        self.crosshair_img = None
        self.generate_crosshair()

        self.movement_threshold = 50
        self.size_ratio_threshold = 0.4

        self.last_command_time = time.time()
        self.command_cooldown = 0.64
        self.prev_command = None

        self.running = True
        self.control_thread = threading.Thread(target=self.control_loop)
        self.control_thread.daemon = True
        self.control_thread.start()

    def generate_crosshair(self):
        self.crosshair_img = np.zeros((40, 40, 4), dtype=np.uint8)
        cv2.line(self.crosshair_img, (20, 0), (20, 40), (0, 255, 0, 255), 1)
        cv2.line(self.crosshair_img, (0, 20), (40, 20), (0, 255, 0, 255), 1)
        cv2.circle(self.crosshair_img, (20, 20), 15, (0, 255, 0, 255), 1)

    def detect_people(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5, False)
        self.net.setInput(blob)
        detections = self.net.forward()
        boxes, confidences = [], []
        for i in range(detections.shape[2]):
            conf = detections[0, 0, i, 2]
            cid = int(detections[0, 0, i, 1])
            if conf > self.conf_threshold and cid == 15:
                box = (detections[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
                x, y, ex, ey = box
                boxes.append((x, y, ex - x, ey - y))
                confidences.append(conf)
        new_people = []
        for x, y, ww, hh in boxes:
            matched = False
            for ox, oy, ow, oh, pid in self.detected_people:
                dist = np.hypot((ox + ow / 2) - (x + ww / 2), (oy + oh / 2) - (y + hh / 2))
                if dist < 100:
                    new_people.append((x, y, ww, hh, pid));
                    matched = True;
                    break
            if not matched:
                new_id = max([p[4] for p in new_people], default=0) + 1
                new_people.append((x, y, ww, hh, new_id))
        self.detected_people = new_people
        if self.active_person_index >= len(new_people): self.active_person_index = -1
        if self.locked_person_index >= len(new_people):
            self.locked_person_index = -1;
            self.tracking = False;
            self.tracking_initialized = False

    def track_locked_person(self, frame):
        if not self.tracking or self.locked_person_index == -1:
            return False

        if not hasattr(self, 'adaptive_tracker'):
            self.adaptive_tracker = AdaptiveMultiTracker()

        if not self.tracking_initialized:
            if self.locked_person_index >= len(self.detected_people):
                self.tracking = False
                self.locked_person_index = -1
                return False

            x, y, ww, hh, _ = self.detected_people[self.locked_person_index]
            detection_conf = 0.8

            success = self.adaptive_tracker.initialize_tracking(frame, (x, y, ww, hh), detection_conf)
            if success:
                self.tracking_initialized = True
                return True
            else:
                self.tracking_initialized = False
                return False

        formatted_detections = []
        for i, (x, y, w, h, person_id) in enumerate(self.detected_people):
            confidence = 0.7
            formatted_detections.append(((x, y, w, h), confidence))

        success, bbox, confidence = self.adaptive_tracker.update(frame, formatted_detections)

        if success and bbox is not None:
            x, y, ww, hh = map(int, bbox)

            cx, cy = x + ww // 2, y + hh // 2
            fx, fy = frame.shape[1] // 2, frame.shape[0] // 2
            offset_x, offset_y = cx - fx, cy - fy
            size_ratio = (ww * hh) / (frame.shape[1] * frame.shape[0])

            state_info = self.adaptive_tracker.get_state_info()

            if confidence > 0.3:
                self.adjust_position(offset_x, offset_y, size_ratio, frame)

            cv2.putText(frame, f"State: {state_info['state']}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f"Confidence: {confidence:.2f}", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            cv2.putText(frame, f"Frames since det: {state_info['frames_since_detection']}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            if state_info['state'] == 'LOST':
                self.tracking_initialized = False
                self.tracking = False
                self.locked_person_index = -1
                return False
            elif state_info['state'] == 'PREDICTING':
                cv2.putText(frame, "PREDICTING TARGET", (10, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            return True
        else:
            self.tracking_initialized = False
            self.tracking = False
            self.locked_person_index = -1
            return False

    def adjust_position(self, offset_x, offset_y, size_ratio, frame):
        print(f"Offset X: {offset_x}, Y: {offset_y}, size_ratio: {size_ratio:.3f}")
        cmd = None
        if abs(offset_x) > self.movement_threshold:
            cmd = 'D' if offset_x > 0 else 'A'
        elif size_ratio < self.size_ratio_threshold * 0.39:
            cmd = 'W'
        if cmd:
            self.send_command(cmd)
            self.send_command(cmd)
            cv2.putText(frame, f"Cmd:{cmd}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def send_command(self, cmd):
        try:
            requests.post("http://192.168.103.41:5000/send", json={"cmd": cmd}, timeout=0.5)
            time.sleep(0.05)
        except Exception as e:
            print(f"[ERROR] Failed to send command to Pi: {e}")

    def switch_active_person(self):
        if not self.detected_people:
            return
        if self.active_person_index == -1:
            self.active_person_index = 0
        else:
            self.active_person_index = (self.active_person_index + 1) % len(self.detected_people)

    def toggle_lock(self):
        if self.active_person_index == -1 or not self.detected_people:
            return
        if self.locked_person_index == -1:
            self.locked_person_index = self.active_person_index
            self.tracking = True
            self.tracking_initialized = False
        else:
            self.locked_person_index = -1
            self.tracking = False
            self.tracking_initialized = False

    def draw_interface(self, frame):
        frame_center_x = frame.shape[1] // 2
        frame_center_y = frame.shape[0] // 2

        cv2.line(frame, (frame_center_x, frame_center_y - 20),
                 (frame_center_x, frame_center_y + 20), (255, 255, 255), 1)
        cv2.line(frame, (frame_center_x - 20, frame_center_y),
                 (frame_center_x + 20, frame_center_y), (255, 255, 255), 1)
        cv2.circle(frame, (frame_center_x, frame_center_y), 15, (255, 255, 255), 1)

        if self.locked_person_index != -1 and self.locked_person_index < len(self.detected_people):
            x, y, w, h, person_id = self.detected_people[self.locked_person_index]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(frame, f"Person {person_id} (LOCKED)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            person_center_x = x + w // 2
            person_center_y = y + h // 2

            cv2.line(frame, (frame_center_x, frame_center_y),
                     (person_center_x, person_center_y), (0, 255, 255), 1)

            offset_x = person_center_x - frame_center_x
            offset_y = person_center_y - frame_center_y
            cv2.putText(frame, f"Offset X: {offset_x}, Y: {offset_y}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        else:
            for i, (x, y, w, h, person_id) in enumerate(self.detected_people):
                color = (0, 255, 0) if i == self.active_person_index else (255, 0, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                status = " (ACTIVE)" if i == self.active_person_index else ""
                cv2.putText(frame, f"Person {person_id}{status}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, "TAB: Switch Target, L: Lock/Unlock", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "WASD: Manual Move, G/H: Gimbal, X: Stop", (10, frame.shape[0] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def control_loop(self):
        while self.running:
            time.sleep(0.2)


tracker = RoverTrackingSystem(camera_index=0,
                              serial_port='COM7',
                              baud_rate=9600)


def gen_frame():
    ret, frame = tracker.cap.read()
    if not ret: return None
    tracker.detect_people(frame)
    if tracker.tracking and tracker.locked_person_index != -1:
        tracker.track_locked_person(frame)
    tracker.draw_interface(frame)
    _, jpg = cv2.imencode('.jpg', frame)
    return jpg.tobytes()