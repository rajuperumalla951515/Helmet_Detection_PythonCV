import cv2
import math
import cvzone
from ultralytics import YOLO
import os
import time
from flask import Flask, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load YOLO model
model = YOLO("weights/best (1).pt")
classNames = ['With Helmet', 'Helmet Notfound']

def generate_frames():
    cap = cv2.VideoCapture(0)
    
    # Frame skipping for performance
    frame_count = 0
    skip_frames = 1 
    last_results = []

    while True:
        success, img = cap.read()
        if not success:
            break
        
        # Resize for better streaming performance
        img = cv2.resize(img, (640, 480))
        
        if frame_count % (skip_frames + 1) == 0:
            results = model(img, stream=True, imgsz=320)
            last_results = list(results)
        
        frame_count += 1

        for r in last_results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h))
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1)

        # Encode the frame in JPEG format
        (flag, encodedImage) = cv2.imencode(".jpg", img)
        if not flag:
            continue
        
        # Yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    # Ensure screenshots directory exists (if needed for background logic)
    if not os.path.exists('screenshots'):
        os.makedirs('screenshots')
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
