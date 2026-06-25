#!/usr/bin/env python3

import cv2
import time
import sys
import threading
import numpy as np
import torch
from flask import Flask, Response, jsonify
from collections import deque
from jetracer.nvidia_racecar import NvidiaRacecar

# =========================
# YOLOv5 LOCAL SETUP
# =========================
sys.path.append("/home/jetson/yolov5")
from models.experimental import attempt_load
from utils.general import non_max_suppression

# =========================
# FLASK + CAR
# =========================
app = Flask(__name__)
car = NvidiaRacecar()

running = True
lock = threading.Lock()

latest_frame = None
display_frame = None
map_frame = np.zeros((600, 600, 3), dtype=np.uint8)

# =========================
# SPEEDS
# =========================
FORWARD_SPEED = 0.15
TURN_SPEED = 0.16
REVERSE_SPEED = -0.18

# =========================
# AI SETTINGS
# =========================
IMG_SIZE = 416
YOLO_EVERY = 0.7
ORB_EVERY = 0.25

# =========================
# LOAD YOLO
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load("/home/jetson/yolov5/yolov5s_v5.pt", map_location=device)
model.eval()

# =========================
# ORB / MEMORY
# =========================
orb = cv2.ORB_create(nfeatures=700)
prev_gray = None
prev_kp = None
prev_des = None

robot_x = 300
robot_y = 300
robot_theta = 0

trajectory = deque(maxlen=300)

scene_memory = {
    "front": "unknown",
    "left": "unknown",
    "right": "unknown",
    "state": "initializing"
}

last_yolo_time = 0
last_orb_time = 0
last_action = "forward"

obstacle_detected = False


# =========================
# CAMERA PIPELINE
# =========================
def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, "
        "format=(string)NV12, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=true max-buffers=1"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


# =========================
# MOTOR FUNCTIONS
# =========================
def stop_car():
    car.throttle = 0.0
    car.steering = 0.0


def move_forward():
    global last_action
    last_action = "forward"
    car.steering = 0.0
    car.throttle = FORWARD_SPEED


def reverse():
    global last_action
    last_action = "reverse"
    car.steering = 0.0
    car.throttle = REVERSE_SPEED


def turn_left():
    global last_action
    last_action = "left"
    car.steering = -0.6
    car.throttle = TURN_SPEED


def turn_right():
    global last_action
    last_action = "right"
    car.steering = 0.6
    car.throttle = TURN_SPEED


# =========================
# YOLO PREPROCESSING
# =========================
def preprocess_yolo(frame):
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device).float()
    img /= 255.0

    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    return img


# =========================
# SIMPLE WALL / OBSTACLE LOGIC
# =========================
def analyze_scene(frame):
    global obstacle_detected

    h, w, _ = frame.shape

    front_roi = frame[int(h * 0.35):int(h * 0.85), int(w * 0.35):int(w * 0.65)]
    left_roi = frame[int(h * 0.35):int(h * 0.85), int(w * 0.05):int(w * 0.30)]
    right_roi = frame[int(h * 0.35):int(h * 0.85), int(w * 0.70):int(w * 0.95)]

    def edge_score(roi):
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 80, 160)
        return np.sum(edges > 0) / edges.size

    front_score = edge_score(front_roi)
    left_score = edge_score(left_roi)
    right_score = edge_score(right_roi)

    front_blocked = front_score > 0.075
    left_blocked = left_score > 0.075
    right_blocked = right_score > 0.075

    scene_memory["front"] = "wall/obstacle" if front_blocked else "free"
    scene_memory["left"] = "wall/obstacle" if left_blocked else "free"
    scene_memory["right"] = "wall/obstacle" if right_blocked else "free"

    if front_blocked and left_blocked and right_blocked:
        scene_memory["state"] = "POSSIBLE_CLOSED_ROOM"
    elif front_blocked:
        scene_memory["state"] = "OBSTACLE_AHEAD"
    elif not front_blocked:
        scene_memory["state"] = "OPEN_PATH"
    else:
        scene_memory["state"] = "UNKNOWN"

    obstacle_detected = front_blocked

    return front_score, left_score, right_score


# =========================
# YOLO DETECTION
# =========================
def run_yolo(frame):
    global obstacle_detected

    img = preprocess_yolo(frame)

    with torch.no_grad():
        pred = model(img)[0]
        pred = non_max_suppression(pred, 0.4, 0.45)[0]

    h, w, _ = frame.shape

    if pred is not None:
        for *xyxy, conf, cls in pred:
            x1, y1, x2, y2 = [int(v.item()) for v in xyxy]

            x1 = int(x1 * w / IMG_SIZE)
            x2 = int(x2 * w / IMG_SIZE)
            y1 = int(y1 * h / IMG_SIZE)
            y2 = int(y2 * h / IMG_SIZE)

            cx = (x1 + x2) // 2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "YOLO obstacle",
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            if int(w * 0.35) < cx < int(w * 0.65):
                obstacle_detected = True
                scene_memory["front"] = "object"
                scene_memory["state"] = "OBJECT_AHEAD"

    return frame


# =========================
# SIMPLE VISUAL ODOMETRY
# =========================
def update_visual_odometry(frame):
    global prev_gray, prev_kp, prev_des
    global robot_x, robot_y, robot_theta

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp, des = orb.detectAndCompute(gray, None)

    if prev_des is not None and des is not None and len(kp) > 10:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(prev_des, des)
        matches = sorted(matches, key=lambda x: x.distance)

        if len(matches) > 15:
            dx = 0
            dy = 0

            for m in matches[:30]:
                p1 = prev_kp[m.queryIdx].pt
                p2 = kp[m.trainIdx].pt
                dx += p2[0] - p1[0]
                dy += p2[1] - p1[1]

            dx /= 30
            dy /= 30

            if last_action == "forward":
                robot_x += int(4 * np.cos(robot_theta))
                robot_y += int(4 * np.sin(robot_theta))

            elif last_action == "left":
                robot_theta -= 0.08

            elif last_action == "right":
                robot_theta += 0.08

            elif last_action == "reverse":
                robot_x -= int(3 * np.cos(robot_theta))
                robot_y -= int(3 * np.sin(robot_theta))

            trajectory.append((robot_x, robot_y))

    prev_gray = gray
    prev_kp = kp
    prev_des = des

    return kp


# =========================
# DRAW MAP
# =========================
def draw_map():
    global map_frame

    m = np.zeros((600, 600, 3), dtype=np.uint8)

    for i in range(1, len(trajectory)):
        cv2.line(m, trajectory[i - 1], trajectory[i], (255, 255, 255), 2)

    cv2.circle(m, (robot_x, robot_y), 8, (0, 0, 255), -1)

    direction_x = int(robot_x + 25 * np.cos(robot_theta))
    direction_y = int(robot_y + 25 * np.sin(robot_theta))
    cv2.line(m, (robot_x, robot_y), (direction_x, direction_y), (0, 255, 255), 2)

    cv2.putText(m, "Scene Memory", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.putText(m, "Front: " + scene_memory["front"], (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(m, "Left: " + scene_memory["left"], (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(m, "Right: " + scene_memory["right"], (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(m, "State: " + scene_memory["state"], (20, 185),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

    map_frame = m


# =========================
# CAMERA THREAD
# =========================
def camera_thread():
    global latest_frame, display_frame
    global last_yolo_time, last_orb_time

    cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        print("Camera failed to open.")
        return

    print("Camera started.")

    while running:
        ret, frame = cap.read()

        if not ret:
            continue

        now = time.time()

        front_score, left_score, right_score = analyze_scene(frame)

        if now - last_yolo_time > YOLO_EVERY:
            frame = run_yolo(frame)
            last_yolo_time = now

        if now - last_orb_time > ORB_EVERY:
            kp = update_visual_odometry(frame)
            frame = cv2.drawKeypoints(frame, kp, None, color=(255, 0, 0))
            last_orb_time = now

        draw_map()

        h, w, _ = frame.shape

        cv2.rectangle(frame, (int(w * 0.35), int(h * 0.35)),
                      (int(w * 0.65), int(h * 0.85)), (0, 255, 255), 2)

        cv2.putText(frame, "Front: " + scene_memory["front"], (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.putText(frame, "State: " + scene_memory["state"], (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.putText(frame, "Action: " + last_action, (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        with lock:
            latest_frame = frame.copy()
            display_frame = frame.copy()

    cap.release()


# =========================
# AUTONOMOUS DECISION THREAD
# =========================
def autonomous_thread():
    while running:
        state = scene_memory["state"]

        if state == "POSSIBLE_CLOSED_ROOM":
            stop_car()
            time.sleep(0.3)
            reverse()
            time.sleep(0.8)
            turn_right()
            time.sleep(0.8)

        elif state == "OBJECT_AHEAD" or state == "OBSTACLE_AHEAD":
            stop_car()
            time.sleep(0.2)
            reverse()
            time.sleep(0.6)

            if scene_memory["left"] == "free":
                turn_left()
            else:
                turn_right()

            time.sleep(0.7)

        else:
            move_forward()
            time.sleep(0.15)


# =========================
# FLASK STREAMING
# =========================
def generate_video():
    while True:
        with lock:
            frame = display_frame.copy() if display_frame is not None else None

        if frame is None:
            continue

        ret, jpeg = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpeg.tobytes() +
            b"\r\n"
        )


def generate_map():
    while True:
        frame = map_frame.copy()

        ret, jpeg = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpeg.tobytes() +
            b"\r\n"
        )


@app.route("/")
def index():
    return """
    <html>
    <head>
        <title>Jetson AI Robot Scene Understanding</title>

        <style>

        body{
            background-color:#111;
            color:white;
            font-family:Arial,sans-serif;
        }

        h1{
            color:#00ff00;
            text-align:center;
        }

        .title{
            color:#00ff00;
            font-size:28px;
            font-weight:bold;
            text-align:center;
            margin-bottom:10px;
        }

        .container{
            display:flex;
            flex-direction:row;
            justify-content:center;
            align-items:flex-start;
            gap:20px;
        }

        .panel{
            text-align:center;
        }

        .panel img{
            border:2px solid #00ff00;
        }

        .status-title{
            color:#00ff00;
            text-align:center;
        }

        #status{
            width:500px;
            margin:auto;
            border:2px solid #00ff00;
            padding:10px;
            font-size:18px;
        }

        </style>

    </head>

    <body>

        <h1>Jetson AI Robot - Monocular Scene Understanding</h1>

        <div class="container">

            <div class="panel">
                <div class="title">
                    Live Camera + YOLO + ORB
                </div>
                <img src="/video" width="700">
            </div>

            <div class="panel">
                <div class="title">
                    Memory Map + Scene Reasoning
                </div>
                <img src="/map" width="700">
            </div>

        </div>

        <h2 class="status-title">Status</h2>
        <pre id="status"></pre>

        <script>
            setInterval(function(){
                fetch('/status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('status').textContent =
                    JSON.stringify(data, null, 2);
                });
            }, 500);
        </script>

    </body>
    </html>
    """

@app.route("/video")
def video():
    return Response(generate_video(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/map")
def map_stream():
    return Response(generate_map(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/status")
def status():
    return jsonify(scene_memory)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    try:
        t1 = threading.Thread(target=camera_thread)
        t2 = threading.Thread(target=autonomous_thread)

        t1.daemon = True
        t2.daemon = True

        t1.start()
        t2.start()

        print("Server running on: http://JETSON_IP:5000")
        app.run(host="0.0.0.0", port=5000, threaded=True)

    except KeyboardInterrupt:
        running = False
        stop_car()
        print("Stopped.")

    finally:
        stop_car()
