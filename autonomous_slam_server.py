import cv2
import time
import threading
import numpy as np
import torch
import sys
from flask import Flask, Response
from jetracer.nvidia_racecar import NvidiaRacecar

sys.path.append("/home/jetson/yolov5")
from models.experimental import attempt_load

app = Flask(__name__)
car = NvidiaRacecar()

running = True
obstacle_detected = False
latest_frame = None
lock = threading.Lock()

FORWARD_SPEED = 0.15
TURN_SPEED = 0.15
REVERSE_SPEED = -0.18

trajectory = np.zeros((480, 480, 3), dtype=np.uint8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load("/home/jetson/yolov5/yolov5s_v5.pt", map_location=device)
model.eval()


def stop_car():
    car.throttle = 0.0
    car.steering = 0.0


def reverse_car():
    print("Reversing")
    car.steering = 0.0
    car.throttle = REVERSE_SPEED
    time.sleep(1.0)
    stop_car()
    time.sleep(0.3)


def turn_away():
    print("Turning away")
    car.steering = 0.7
    car.throttle = TURN_SPEED
    time.sleep(1.2)
    stop_car()
    time.sleep(0.3)


def autonomous_drive():
    global running, obstacle_detected

    try:
        while running:
            with lock:
                obs = obstacle_detected

            if obs:
                print("Obstacle detected! Stop and change direction")
                stop_car()
                time.sleep(0.5)
                reverse_car()
                turn_away()
            else:
                print("Moving forward slowly")
                car.steering = 0.0
                car.throttle = FORWARD_SPEED
                time.sleep(0.2)

    finally:
        stop_car()


def detect_obstacle_yolo(frame):
    h, w, _ = frame.shape

    roi_x1 = int(w * 0.30)
    roi_x2 = int(w * 0.70)
    roi_y1 = int(h * 0.25)
    roi_y2 = int(h * 0.85)

    obstacle = False

    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)[0]

    pred = pred.cpu().numpy()

    for det in pred[0]:
        conf = det[4]

        if conf < 0.4:
            continue

        x1, y1, x2, y2 = det[0], det[1], det[2], det[3]

        x1 = int(x1 * w / 640)
        y1 = int(y1 * h / 640)
        x2 = int(x2 * w / 640)
        y2 = int(y2 * h / 640)

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        inside_roi = (
            roi_x1 < center_x < roi_x2 and
            roi_y1 < center_y < roi_y2
        )

        if inside_roi:
            obstacle = True
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.rectangle(
        frame,
        (roi_x1, roi_y1),
        (roi_x2, roi_y2),
        (0, 255, 255),
        2
    )

    return obstacle, frame


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

if not ret:
    print("Camera not working")
    stop_car()
    exit()

prev_frame = cv2.resize(prev_frame, (640, 480))
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
kp1, des1 = orb.detectAndCompute(prev_gray, None)

x, y = 240, 240


def camera_loop():
    global running, obstacle_detected, latest_frame
    global prev_gray, kp1, des1
    global x, y, trajectory

    while running:
        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.resize(frame, (640, 480))

        obs, frame = detect_obstacle_yolo(frame)

        with lock:
            obstacle_detected = obs

        print("Obstacle:", obs)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = orb.detectAndCompute(gray, None)

        if des1 is not None and des2 is not None:
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

        kp1, des1 = kp2, des2
        prev_gray = gray

        if obs:
            cv2.putText(frame, "OBSTACLE DETECTED", (120, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        else:
            cv2.putText(frame, "CLEAR", (250, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        combined = np.hstack((frame, trajectory))

        cv2.putText(combined, "YOLO Obstacle Detection", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(combined, "ORB Trajectory Map", (700, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        with lock:
            latest_frame = combined.copy()

        time.sleep(0.05)


def generate_frames():
    global latest_frame

    while True:
        with lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            continue

        ret, buffer = cv2.imencode(".jpg", frame)

        if not ret:
            continue

        jpg = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
        )


@app.route("/")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


try:
    camera_thread = threading.Thread(target=camera_loop)
    camera_thread.daemon = True
    camera_thread.start()

    drive_thread = threading.Thread(target=autonomous_drive)
    drive_thread.daemon = True
    drive_thread.start()

    app.run(host="0.0.0.0", port=5000)

except KeyboardInterrupt:
    print("Stopping system...")

finally:
    running = False
    stop_car()
    cap.release()
