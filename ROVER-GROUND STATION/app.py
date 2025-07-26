from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from datetime import datetime
import threading
import time

ROVER_PIN = 18
FIRSTAID_PIN = 19

try:
    import pigpio
    pi = pigpio.pi()
    if not pi.connected:
        raise Exception("pigpio daemon not running")
    PIGPIO_AVAILABLE = True
    print("pigpio initialized")
except Exception as e:
    print("pigpio not available:", e)
    pi = None
    PIGPIO_AVAILABLE = False

app = Flask(__name__)
app.config['SECRET_KEY'] = 'emergency_system_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

current_alerts = {
    'emergency': False,
    'emergency_time': None,
    'reports': [],
    'rover_launching': False,
    'rover_launch_time': None,
    'firstaid_dropping': False,
    'firstaid_drop_time': None
}

@app.route('/')
def control_panel():
    return render_template('control_panel.html')

@app.route('/monitor')
def monitoring_dashboard():
    return render_template('monitor.html')

@app.route('/api/status')
def get_status():
    return jsonify(current_alerts)

@app.route('/emergency', methods=['POST'])
def emergency_alert():
    current_alerts['emergency'] = True
    current_alerts['emergency_time'] = datetime.now().strftime('%H:%M:%S')
    socketio.emit('emergency_alert', {
        'message': 'EMERGENCY ON STATION 1',
        'time': current_alerts['emergency_time']
    })
    return jsonify({'status': 'Emergency alert sent'})

@app.route('/request_firstaid', methods=['POST'])
def request_firstaid():
    now = datetime.now().strftime('%H:%M:%S')
    socketio.emit('firstaid_requested', {
        'message': 'REQUESTED FIRST AID ON STATION 1',
        'time': now
    })
    return jsonify({'status': 'First aid request sent'})

@app.route('/clear_emergency', methods=['POST'])
def clear_emergency():
    current_alerts['emergency'] = False
    current_alerts['emergency_time'] = None
    socketio.emit('emergency_cleared')
    return jsonify({'status': 'Emergency cleared'})

@app.route('/report', methods=['POST'])
def submit_report():
    report_text = request.json.get('report', '')
    if report_text.strip():
        report_data = {
            'text': report_text,
            'time': datetime.now().strftime('%H:%M:%S'),
            'date': datetime.now().strftime('%Y-%m-%d')
        }
        current_alerts['reports'].append(report_data)
        socketio.emit('new_report', {
            'report': report_data,
            'station': 'Station 1'
        })
    return jsonify({'status': 'Report submitted'})

def control_rover_servo():
    if PIGPIO_AVAILABLE:
        try:
            print("ROVER: 1200 µs → 2.2s")
            pi.set_servo_pulsewidth(ROVER_PIN, 1200)
            time.sleep(2.2)

            print("ROVER: 1500 µs → 3s")
            pi.set_servo_pulsewidth(ROVER_PIN, 1500)
            time.sleep(3)

            print("ROVER: 1800 µs → 2s")
            pi.set_servo_pulsewidth(ROVER_PIN, 1800)
            time.sleep(2)

            print("ROVER: 1500 µs → 2s")
            pi.set_servo_pulsewidth(ROVER_PIN, 1500)
            time.sleep(2)

            print("ROVER: stopping")
            pi.set_servo_pulsewidth(ROVER_PIN, 0)
        except Exception as e:
            print(f"ROVER servo error: {e}")
            pi.set_servo_pulsewidth(ROVER_PIN, 0)
    else:
        print("Simulating rover servo")

def control_firstaid_servo():
    if PIGPIO_AVAILABLE:
        try:
            print("FIRSTAID: 900 µs → 2.2s")
            pi.set_servo_pulsewidth(FIRSTAID_PIN, 900)
            time.sleep(2)

            print("FIRSTAID: 1200 µs → 5s")
            pi.set_servo_pulsewidth(FIRSTAID_PIN, 1200)
            time.sleep(5)

            print("FIRSTAID: 1500 µs → 2s")
            pi.set_servo_pulsewidth(FIRSTAID_PIN, 1500)
            time.sleep(2)

            print("FIRSTAID: 1200 µs → 2s")
            pi.set_servo_pulsewidth(FIRSTAID_PIN, 1200)
            time.sleep(2)

            print("FIRSTAID: stopping")
            pi.set_servo_pulsewidth(FIRSTAID_PIN, 0)
        except Exception as e:
            print(f"FIRSTAID servo error: {e}")
            pi.set_servo_pulsewidth(FIRSTAID_PIN, 0)
    else:
        print("Simulating firstaid servo")

def launch_rover_sequence():
    current_alerts['rover_launching'] = True
    current_alerts['rover_launch_time'] = datetime.now().strftime('%H:%M:%S')
    socketio.emit('rover_launching', {
        'message': 'LAUNCHING ROVER',
        'time': current_alerts['rover_launch_time']
    })
    control_rover_servo()
    time.sleep(2)
    current_alerts['rover_launching'] = False
    current_alerts['rover_launch_time'] = None
    socketio.emit('rover_launch_complete')

@app.route('/launch_rover', methods=['POST'])
def launch_rover():
    if not current_alerts['rover_launching']:
        thread = threading.Thread(target=launch_rover_sequence, daemon=True)
        thread.start()
        return jsonify({'status': 'Rover launch initiated'})
    else:
        return jsonify({'status': 'Rover launch already in progress'})

@app.route('/drop_firstaid', methods=['POST'])
def drop_firstaid():
    if not current_alerts['firstaid_dropping']:
        current_alerts['firstaid_dropping'] = True
        current_alerts['firstaid_drop_time'] = datetime.now().strftime('%H:%M:%S')
        socketio.emit('firstaid_dropping', {
            'message': 'DROPPING FIRSTAID',
            'time': current_alerts['firstaid_drop_time']
        })

        def drop_sequence():
            control_firstaid_servo()
            time.sleep(2)
            current_alerts['firstaid_dropping'] = False
            current_alerts['firstaid_drop_time'] = None
            socketio.emit('firstaid_drop_complete')

        threading.Thread(target=drop_sequence, daemon=True).start()
        return jsonify({'status': 'Firstaid drop initiated'})
    else:
        return jsonify({'status': 'Firstaid drop already in progress'})

def auto_clear_emergency():
    emergency_start = None
    while True:
        if current_alerts['emergency']:
            if emergency_start is None:
                emergency_start = time.time()
            elif time.time() - emergency_start >= 300:
                current_alerts['emergency'] = False
                current_alerts['emergency_time'] = None
                emergency_start = None
                socketio.emit('emergency_cleared')
        else:
            emergency_start = None
        time.sleep(1)

threading.Thread(target=auto_clear_emergency, daemon=True).start()

import atexit
@atexit.register
def shutdown():
    if PIGPIO_AVAILABLE and pi:
        pi.set_servo_pulsewidth(ROVER_PIN, 0)
        pi.set_servo_pulsewidth(FIRSTAID_PIN, 0)
        pi.stop()
        print("pigpio cleaned up")

if __name__ == '__main__':
    print("Control Panel: http://localhost:5000")
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
