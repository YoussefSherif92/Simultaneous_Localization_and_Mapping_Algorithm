#!/usr/bin/env python3
import cv2
import time
import threading
import numpy as np
import torch
import sys
from collections import deque
from flask import Flask, Response
from jetracer.nvidia_racecar import NvidiaRacecar

sys.path.append("/home/jetson/yolov5")
from models.experimental import attempt_load
from utils.general import non_max_suppression

app = Flask(__name__)
car = NvidiaRacecar()

running = True
obstacle_detected = False
latest_frame = None
lock = threading.Lock()

FORWARD_SPEED = 0.15
TURN_SPEED = 0.16
REVERSE_SPEED = -0.18

YOLO_EVERY = 0.70
IMG_SIZE = 416

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = attempt_load("/home/jetson/yolov5/yolov5s_v5.pt", map_location=device)
model.eval()

orb = cv2.ORB_create(nfeatures=1500, fastThreshold=7)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

MAP_SIZE = 600
slam_map = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)

pose_x = MAP_SIZE // 2
pose_y = MAP_SIZE // 2
pose_theta = 0.0

pose_history = deque(maxlen=8)
keyframes = []
loop_closed_count = 0

poses = []
edges = []
optimized_poses = []

prev_gray = None
prev_kp = None
prev_des = None

last_yolo_time = 0
last_obstacle = False


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
    time.sleep(1.1)
    stop_car()
    time.sleep(0.3)


def autonomous_drive():
    global running

    try:
        while running:
            with lock:
                obs = obstacle_detected

            if obs:
                print("Obstacle detected! Stop -> reverse -> turn")
                stop_car()
                time.sleep(0.4)
                reverse_car()
                turn_away()
            else:
                car.steering = 0.0
                car.throttle = FORWARD_SPEED
                time.sleep(0.15)

    finally:
        stop_car()


def optimize_pose_graph(poses, edges, iterations=40, lr=0.008):
    if len(poses) < 3:
        return poses

    opt = np.array(poses, dtype=np.float32)

    for _ in range(iterations):
        grad = np.zeros_like(opt)

        for i, j, dx, dy, weight in edges:
            if i >= len(opt) or j >= len(opt):
                continue

            predicted_dx = opt[j, 0] - opt[i, 0]
            predicted_dy = opt[j, 1] - opt[i, 1]

            error_x = predicted_dx - dx
            error_y = predicted_dy - dy

            grad[i, 0] += weight * error_x
            grad[i, 1] += weight * error_y

            grad[j, 0] -= weight * error_x
            grad[j, 1] -= weight * error_y

        opt -= lr * grad
        opt[0] = poses[0]

    return opt.tolist()


def detect_obstacle_yolo(frame):
    h, w, _ = frame.shape

    roi_x1 = int(w * 0.30)
    roi_x2 = int(w * 0.70)
    roi_y1 = int(h * 0.25)
    roi_y2 = int(h * 0.90)

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose((2, 0, 1))
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device).float() / 255.0
    img = img.unsqueeze(0)

    with torch.no_grad():
        pred = model(img)[0]

    pred = non_max_suppression(pred, conf_thres=0.40, iou_thres=0.45)

    obstacle = False

    for det in pred:
        if det is None or len(det) == 0:
            continue

        for *xyxy, conf, cls in det:
            x1, y1, x2, y2 = xyxy

            x1 = int(x1.item() * w / IMG_SIZE)
            y1 = int(y1.item() * h / IMG_SIZE)
            x2 = int(x2.item() * w / IMG_SIZE)
            y2 = int(y2.item() * h / IMG_SIZE)

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            inside_roi = roi_x1 < cx < roi_x2 and roi_y1 < cy < roi_y2

            if inside_roi:
                obstacle = True
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            cv2.circle(frame, (cx, cy), 3, color, -1)

    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 255), 1)

    return obstacle, frame


def update_visual_odometry(gray):
    global prev_gray, prev_kp, prev_des
    global pose_x, pose_y, pose_theta
    global keyframes, loop_closed_count
    global poses, edges, optimized_poses

    kp, des = orb.detectAndCompute(gray, None)

    if prev_gray is None or prev_des is None or des is None or len(kp) < 20:
        prev_gray = gray
        prev_kp = kp
        prev_des = des
        return 0, 0, 0, 0, False

    matches = bf.match(prev_des, des)
    matches = sorted(matches, key=lambda m: m.distance)
    good = matches[:70]

    if len(good) < 15:
        prev_gray = gray
        prev_kp = kp
        prev_des = des
        return 0, 0, 0, len(good), False

    pts1 = np.float32([prev_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts2 = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, inliers = cv2.estimateAffinePartial2D(
        pts1,
        pts2,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0
    )

    if M is None:
        prev_gray = gray
        prev_kp = kp
        prev_des = des
        return 0, 0, 0, len(good), False

    dx_img = M[0, 2]
    dy_img = M[1, 2]
    dtheta = np.arctan2(M[1, 0], M[0, 0])

    scale = 0.45

    forward = -dy_img * scale
    side = -dx_img * scale

    pose_theta += dtheta * 0.5

    dx_world = forward * np.cos(pose_theta) - side * np.sin(pose_theta)
    dy_world = forward * np.sin(pose_theta) + side * np.cos(pose_theta)

    pose_x += dx_world
    pose_y += dy_world

    pose_history.append((pose_x, pose_y, pose_theta))

    pose_x = np.mean([p[0] for p in pose_history])
    pose_y = np.mean([p[1] for p in pose_history])
    pose_theta = np.mean([p[2] for p in pose_history])

    pose_x = int(max(20, min(MAP_SIZE - 20, pose_x)))
    pose_y = int(max(20, min(MAP_SIZE - 20, pose_y)))

    poses.append([pose_x, pose_y, pose_theta])

    if len(poses) > 350:
        poses = poses[-350:]
        edges = []
        keyframes = []

    if len(poses) > 1:
        i = len(poses) - 2
        j = len(poses) - 1
        dx_pose = poses[j][0] - poses[i][0]
        dy_pose = poses[j][1] - poses[i][1]
        edges.append((i, j, dx_pose, dy_pose, 1.0))

    loop_closed = False

    if len(keyframes) == 0:
        keyframes.append((pose_x, pose_y, pose_theta, des, len(poses) - 1))
    else:
        last_kx, last_ky, _, _, _ = keyframes[-1]
        dist_from_last = np.hypot(pose_x - last_kx, pose_y - last_ky)

        if dist_from_last > 35:
            keyframes.append((pose_x, pose_y, pose_theta, des, len(poses) - 1))

        if len(keyframes) > 8 and des is not None:
            for old_kx, old_ky, old_th, old_des, old_pose_index in keyframes[:-5]:
                if old_des is None:
                    continue

                old_matches = bf.match(old_des, des)
                old_matches = sorted(old_matches, key=lambda m: m.distance)

                close_in_image = len([m for m in old_matches[:50] if m.distance < 45])
                close_in_map = np.hypot(pose_x - old_kx, pose_y - old_ky)

                if close_in_image > 25 and close_in_map < 120:
                    current_pose_index = len(poses) - 1

                    loop_dx = pose_x - old_kx
                    loop_dy = pose_y - old_ky

                    edges.append((old_pose_index, current_pose_index, loop_dx, loop_dy, 5.0))

                    optimized_poses[:] = optimize_pose_graph(poses, edges)

                    if len(optimized_poses) > 0:
                        pose_x = int(optimized_poses[-1][0])
                        pose_y = int(optimized_poses[-1][1])
                        pose_theta = optimized_poses[-1][2]

                    loop_closed = True
                    loop_closed_count += 1
                    break

    prev_gray = gray
    prev_kp = kp
    prev_des = des

    return dx_img, dy_img, dtheta, len(good), loop_closed


def draw_slam_map(obstacle):
    global slam_map

    slam_map[:] = (slam_map * 0.985).astype(np.uint8)

    px = int(pose_x)
    py = int(pose_y)

    front_x = int(px + 35 * np.cos(pose_theta))
    front_y = int(py + 35 * np.sin(pose_theta))

    if len(poses) > 1:
        for i in range(1, len(poses)):
            p1 = poses[i - 1]
            p2 = poses[i]
            cv2.line(slam_map, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (0, 120, 0), 1)

    if len(optimized_poses) > 1:
        for i in range(1, len(optimized_poses)):
            p1 = optimized_poses[i - 1]
            p2 = optimized_poses[i]
            cv2.line(slam_map, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 255), 2)

    if obstacle:
        ox = int(px + 70 * np.cos(pose_theta))
        oy = int(py + 70 * np.sin(pose_theta))
        cv2.circle(slam_map, (ox, oy), 12, (0, 0, 255), -1)
        cv2.line(slam_map, (px, py), (ox, oy), (0, 0, 180), 1)

    for kx, ky, _, _, _ in keyframes:
        cv2.circle(slam_map, (int(kx), int(ky)), 4, (255, 255, 0), 1)

    cv2.circle(slam_map, (px, py), 8, (0, 255, 255), 2)
    cv2.line(slam_map, (px, py), (front_x, front_y), (0, 255, 255), 2)

    return slam_map.copy()


gst = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=640, height=480, format=NV12, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=320, height=240, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink drop=true max-buffers=1 sync=false"
)

cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Camera not opened")
    stop_car()
    exit()


def camera_loop():
    global running, obstacle_detected, latest_frame
    global last_yolo_time, last_obstacle

    while running:
        ret, frame = cap.read()

        if not ret:
            continue

        frame = cv2.resize(frame, (320, 240))

        now = time.time()

        if now - last_yolo_time > YOLO_EVERY:
            obs, frame = detect_obstacle_yolo(frame)
            last_obstacle = obs
            last_yolo_time = now
        else:
            obs = last_obstacle

        with lock:
            obstacle_detected = obs

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dx, dy, dtheta, matches, loop_closed = update_visual_odometry(gray)

        map_view = draw_slam_map(obs)

        if obs:
            status = "OBSTACLE"
            color = (0, 0, 255)
        else:
            status = "CLEAR"
            color = (0, 255, 0)

        cv2.putText(frame, status, (90, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
        cv2.putText(frame, "YOLO + ORB SLAM", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.putText(frame, "Matches: {}".format(matches), (10, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        cv2.putText(frame, "dx={:.1f} dy={:.1f}".format(dx, dy), (10, 235), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

        cv2.putText(map_view, "ORB VO + Keyframes + Pose Graph", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        cv2.putText(map_view, "Keyframes:{} Loops:{} Edges:{}".format(len(keyframes), loop_closed_count, len(edges)),
                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        cv2.putText(map_view, "Green raw path | Pink optimized path", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

        if loop_closed:
            cv2.putText(map_view, "LOOP CLOSURE + POSE GRAPH OPTIMIZATION", (35, 560),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 0, 255), 2)

        frame_big = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
        map_big = cv2.resize(map_view, (640, 480), interpolation=cv2.INTER_LINEAR)

        combined = np.hstack((frame_big, map_big))

        with lock:
            latest_frame = combined.copy()

        time.sleep(0.02)


def generate_frames():
    global latest_frame

    while True:
        with lock:
            frame = None if latest_frame is None else latest_frame.copy()

        if frame is None:
            time.sleep(0.02)
            continue

        ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])

        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


@app.route("/")
def video():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


try:
    camera_thread = threading.Thread(target=camera_loop)
    camera_thread.daemon = True
    camera_thread.start()

    drive_thread = threading.Thread(target=autonomous_drive)
    drive_thread.daemon = True
    drive_thread.start()

    app.run(host="0.0.0.0", port=5000, threaded=True)

except KeyboardInterrupt:
    print("Stopping system...")

finally:
    running = False
    stop_car()
    cap.release()
