from flask import Flask, render_template, Response, request, jsonify
from tracking import Tracker
import requests
import logging
import sys
from io import StringIO
import threading
from collections import deque
import json
import time

app = Flask(__name__)

ESP32_IP = "http://192.168.1.50"


class LogCapture:
    def __init__(self, max_logs=100):
        self.logs = deque(maxlen=max_logs)
        self.lock = threading.Lock()

    def add_log(self, message, level='info'):
        with self.lock:
            self.logs.append({
                'message': message,
                'level': level,
                'timestamp': time.time()
            })

    def get_new_logs(self, last_timestamp=0):
        with self.lock:
            return [log for log in self.logs if log['timestamp'] > last_timestamp]


log_capture = LogCapture()

original_print = print


def captured_print(*args, **kwargs):
    message = ' '.join(str(arg) for arg in args)
    log_capture.add_log(message, 'info')
    original_print(*args, **kwargs)


print = captured_print


class WebLogHandler(logging.Handler):
    def __init__(self, log_capture):
        super().__init__()
        self.log_capture = log_capture

    def emit(self, record):
        level_map = {
            'DEBUG': 'info',
            'INFO': 'info',
            'WARNING': 'warn',
            'ERROR': 'error',
            'CRITICAL': 'error'
        }
        level = level_map.get(record.levelname, 'info')
        self.log_capture.add_log(record.getMessage(), level)


logging.basicConfig(level=logging.INFO)
web_handler = WebLogHandler(log_capture)
logging.getLogger().addHandler(web_handler)


def send_command_to_esp32(cmd):
    try:
        requests.post(f"{ESP32_IP}/cmd", data={"value": cmd}, timeout=0.5)
    except Exception as e:
        print(f"Failed to send command to ESP32: {e}")


print("Loading YOLOv8 model inside Tracker...")
print("MediaPipe Pose initialized")
print("MediaPipe Hands initialized")
print("ESP32 connection configured")
print("Starting threat detection system...")

tracker = Tracker(esp32_url=ESP32_IP, send_command_fn=send_command_to_esp32)


@app.route('/')
def index():
    return render_template('drone.html')


@app.route('/video_feed')
def video_feed():
    return Response(tracker.gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/toggle_active', methods=['POST'])
def toggle_active():
    tracker.toggle_active()
    return ('', 204)


@app.route('/toggle_lock', methods=['POST'])
def toggle_lock():
    tracker.toggle_lock()
    return ('', 204)


@app.route('/current_threat')
def current_threat():
    data = tracker.get_current_threat()
    return jsonify(data)


@app.route('/terminal_logs')
def terminal_logs():
    last_timestamp = float(request.args.get('since', 0))
    new_logs = log_capture.get_new_logs(last_timestamp)
    return jsonify({'logs': new_logs})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, threaded=True)