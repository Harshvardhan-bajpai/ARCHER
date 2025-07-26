from flask import Flask, Response, request, render_template
import requests
import rover_tracking

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def frame_generator():
    while True:
        frame = rover_tracking.gen_frame()
        if frame is None:
            continue
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )

@app.route('/video_feed')
def video_feed():
    return Response(
        frame_generator(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

@app.route('/video_feed2')
def video_feed2():
    remote_url = 'http://192.168.103.41:8080/video'
    def generate():
        with requests.get(remote_url, stream=True) as r:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    yield chunk
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/key', methods=['POST'])
def key():
    data = request.get_json()
    key = data.get('key', '')

    if key == 'Tab':
        rover_tracking.tracker.switch_active_person()
        return ('', 204)

    if key.upper() == 'L':
        rover_tracking.tracker.toggle_lock()
        return ('', 204)

    k = key.upper()
    if k in ['W', 'A', 'S', 'D', 'G', 'H', 'X', 'F']:
        rover_tracking.tracker.send_command(k)
        return ('', 204)

    return ('', 204)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)