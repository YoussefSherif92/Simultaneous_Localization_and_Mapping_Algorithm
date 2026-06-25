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
from utils.general import non_max_suppression

app = Flask(__name__)
car = NvidiaRacecar()

running = True

lock = threading.Lock()

latest_frame = None
raw_frame = None
display_frame = None

obstacle_detected = False

FORWARD_SPEED = 0.15
TURN_SPEED = 0.16
REVERSE_SPEED = -0.18

YOLO_EVERY_SECONDS = 0.25
ORB_EVERY_SECONDS = 0.20
STREAM_DELAY = 0.10

trajectory = np.zeros((480, 480, 3), dtype=np.uint8)

x, y = 240, 240

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Use yolov5n if possible. It is faster than yolov5s.
MODEL_PATH = "/home/jetson/yolov5/yolov5s_v5.pt"# MODEL_PATH = "/home/jetson/yolov5/yolov5s_v5.pt"

model = attempt_load(MODEL_PATH, map_location=device)
model.eval()

orb = cv2.ORB_create(nfeatures=300, fastThreshold=20)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

prev_gray = None
kp1 = None
des1 = None


def stop_car():
    car.throttle = 0.0
    car.steering = 0.0


def move_forward():
    car.steering = 0.0
    car.throttle = FORWARD_SPEED


def reverse_car():
    print("Reverse")
    stop_car()
    time.sleep(0.3)

    car.steering = 0.0
    car.throttle = REVERSE_SPEED
    time.sleep(1.0)

    stop_car()
    time.sleep(0.3)


def turn_away():
    print("Turn away")
    stop_car()
    time.sleep(0.2)

    car.steering = 0.7
    car.throttle = TURN_SPEED
    time.sleep(1.2)

    stop_car()
    time.sleep(0.3)


def autonomous_drive():
    global running, obstacle_detected

    while running:
        with lock:
            obs = obstacle_detected

        if obs:
            print("Obstacle detected")
            stop_car()
            time.sleep(0.3)
            reverse_car()
            turn_away()

            with lock:
                obstacle_detected = False

        else:
            move_forward()
            time.sleep(0.1)

    stop_car()


def detect_obstacle_yolo(frame):
    h, w, _ = frame.shape

    roi_x1 = int(w * 0.30)
    roi_x2 = int(w * 0.70)
    roi_y1 = int(h * 0.25)
    roi_y2 = int(h * 0.85)

    obstacle = False
    output = frame.copy()

    img_size = 320

    img = cv2.resize(frame, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0
    img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)[0]

    pred = non_max_suppression(pred, conf_thres=0.40, iou_thres=0.45)

    if pred[0] is not None:
        for det in pred[0]:
            x1, y1, x2, y2, conf, cls = det

            x1 = int(x1.item() * w / img_size)
            y1 = int(y1.item() * h / img_size)
            x2 = int(x2.item() * w / img_size)
            y2 = int(y2.item() * h / img_size)

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

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                output,
                f"{conf:.2f}",
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    cv2.rectangle(output, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 2)

    if obstacle:
        cv2.putText(output, "OBSTACLE", (160, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    else:
        cv2.putText(output, "CLEAR", (230, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    return obstacle, output


def camera_loop():
    global running, raw_frame

    gst = (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width=640, height=480, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink drop=true max-buffers=1"
    )

    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Camera not opened")
        running = False
        stop_car()
        return

    while running:
        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.resize(frame, (640, 480))

        with lock:
            raw_frame = frame.copy()

        time.sleep(0.01)

    cap.release()


def yolo_loop():
    global running, raw_frame, obstacle_detected, latest_frame

    while running:
        with lock:
            frame = None if raw_frame is None else raw_frame.copy()

        if frame is None:
            time.sleep(0.05)
            continue

        obs, yolo_frame = detect_obstacle_yolo(frame)

        with lock:
            obstacle_detected = obs
            latest_frame = yolo_frame.copy()

        time.sleep(YOLO_EVERY_SECONDS)


def orb_mapping_loop():
    global running, raw_frame, trajectory
    global prev_gray, kp1, des1
    global x, y

    while running:
        with lock:
            frame = None if raw_frame is None else raw_frame.copy()

        if frame is None:
            time.sleep(0.05)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = orb.detectAndCompute(gray, None)

        if prev_gray is not None and des1 is not None and des2 is not None:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda m: m.distance)[:30]

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

                x -= int(dx * 1.2)
                y -= int(dy * 1.2)

                x = max(0, min(479, x))
                y = max(0, min(479, y))

                cv2.circle(trajectory, (x, y), 2, (0, 255, 0), -1)
                cv2.circle(trajectory, (x, y), 5, (0, 0, 255), 1)

        prev_gray = gray
        kp1, des1 = kp2, des2

        time.sleep(ORB_EVERY_SECONDS)


def display_loop():
    global running, latest_frame, trajectory, display_frame

    while running:
        with lock:
            frame = None if latest_frame is None else latest_frame.copy()
            map_img = trajectory.copy()

        if frame is None:
            time.sleep(0.05)
            continue

        combined = np.hstack((frame, map_img))

        cv2.putText(combined, "YOLO Detection", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.putText(combined, "ORB Trajectory", (700, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        with lock:
            display_frame = combined.copy()

        time.sleep(STREAM_DELAY)


def generate_frames():
    global display_frame

    while True:
        with lock:
            frame = None if display_frame is None else display_frame.copy()

        if frame is None:
            time.sleep(0.05)
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
    threads = [
        threading.Thread(target=camera_loop, daemon=True),
        threading.Thread(target=yolo_loop, daemon=True),
        threading.Thread(target=orb_mapping_loop, daemon=True),
        threading.Thread(target=display_loop, daemon=True),
        threading.Thread(target=autonomous_drive, daemon=True),
    ]

    for t in threads:
        t.start()

    app.run(host="0.0.0.0", port=5000, threaded=True)

except KeyboardInterrupt:
    print("Stopping system...")

finally:
    running = False
    stop_car()
    time.sleep(0.5)
