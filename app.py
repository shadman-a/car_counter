from flask import Flask, render_template, Response, jsonify
import cv2
from vision import process_frame

app = Flask(__name__)

cap = cv2.VideoCapture(1)
current_counts = {'car': 0, 'person': 0, 'bicycle': 0, 'bus': 0, 'truck': 0}

def generate_frames():
    global current_counts
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame, counts = process_frame(frame)  # Process the frame using the vision module
            current_counts = counts  # Update global count state
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/counts')
def counts():
    return jsonify(current_counts)

if __name__ == "__main__":
    app.run(debug=True)
