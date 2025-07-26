import serial
import cv2
import threading
import time
from flask import Flask, request, Response
import json

try:
    ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1) #arduino com port
    print("Serial connection established")
except Exception as e:
    print(f"Serial connection failed: {e}")
    ser = None

app = Flask(__name__)

camera = None
camera_lock = threading.Lock()
latest_frame = None


def initialize_camera():
    global camera
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            print("Error: Could not open camera")
            return False

        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        print("Camera initialized successfully")
        return True
    except Exception as e:
        print(f"Camera initialization failed: {e}")
        return False


def capture_frames():
    global latest_frame, camera

    while True:
        if camera is None or not camera.isOpened():
            time.sleep(1)
            continue

        try:
            with camera_lock:
                ret, frame = camera.read()
                if ret:
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    latest_frame = buffer.tobytes()
                else:
                    print("Failed to read frame")
                    time.sleep(0.1)
        except Exception as e:
            print(f"Frame capture error: {e}")
            time.sleep(1)

        time.sleep(1 / 30)


@app.route('/send', methods=['POST'])
def send_command():
    data = request.get_json()
    cmd = data.get('cmd', '')
    if cmd:
        ser.write(f"{cmd}\n".encode())
        return f"Sent: {cmd}", 200
    return "No command", 400


@app.route('/get_frame')
def get_frame():
    global latest_frame

    if latest_frame is None:
        return "No frame available", 404

    with camera_lock:
        frame_data = latest_frame

    return Response(frame_data, mimetype='image/jpeg')


@app.route('/camera_status')
def camera_status():
    status = {
        'camera_available': camera is not None and camera.isOpened(),
        'serial_available': ser is not None,
        'latest_frame_available': latest_frame is not None
    }
    return json.dumps(status)


if __name__ == '__main__':
    if initialize_camera():
        capture_thread = threading.Thread(target=capture_frames, daemon=True)
        capture_thread.start()
        print("Camera capture thread started")
    else:
        print("Warning: Camera not available")

    print("Starting server on port 5000...")
    app.run(host='0.0.0.0', port=5000)