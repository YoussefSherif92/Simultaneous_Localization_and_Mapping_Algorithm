from flask import Flask, Response
import cv2
import numpy as np

app = Flask(__name__)

gst = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=640, height=480, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink"
)

cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)

orb = cv2.ORB_create(nfeatures=1500, fastThreshold=5)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
kp1, des1 = orb.detectAndCompute(prev_gray, None)

x, y = 300, 300
trajectory = np.zeros((480, 480, 3), dtype=np.uint8)

def generate_frames():
    global prev_gray, kp1, des1, x, y, trajectory

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = orb.detectAndCompute(gray, None)

        if des1 is not None and des2 is not None and len(kp1) > 10 and len(kp2) > 10:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda m: m.distance)[:50]

            dx_list = []
            dy_list = []

            for m in matches:
                p1 = kp1[m.queryIdx].pt
                p2 = kp2[m.trainIdx].pt
                dx_list.append(p2[0] - p1[0])
                dy_list.append(p2[1] - p1[1])

            if len(dx_list) > 0:
                dx = np.mean(dx_list)
                dy = np.mean(dy_list)

                x -= int(dx * 1.5)
                y -= int(dy * 1.5)

                x = max(0, min(479, x))
                y = max(0, min(479, y))

                cv2.circle(trajectory, (x, y), 2, (0, 255, 0), -1)
                cv2.circle(trajectory, (x, y), 5, (0, 0, 255), 1)

        prev_gray = gray
        kp1, des1 = kp2, des2

        map_resized = cv2.resize(trajectory, (480, 480))
        combined = np.hstack((frame, map_resized))

        cv2.putText(combined, "Live Camera", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(combined, "Robot Trajectory Map", (670, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        ret, buffer = cv2.imencode(".jpg", combined)
        jpg = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")

@app.route("/")
def video():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

app.run(host="0.0.0.0", port=5000)
