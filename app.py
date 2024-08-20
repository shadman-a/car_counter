from flask import Flask, render_template, Response
import cv2
from vision import process_frame  # Import the vision functions

app = Flask(__name__)

# Initialize the webcam
cap = cv2.VideoCapture(1)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            frame, car_count = process_frame(frame)  # Process the frame using the vision module
            cv2.putText(frame, f'Car Count: {car_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)
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

if __name__ == "__main__":
    app.run(debug=True)
