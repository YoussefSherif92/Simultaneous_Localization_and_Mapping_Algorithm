from flask import Flask, Response
import cv2
import sys
import torch

sys.path.append('/home/jetson/yolov5')

from models.experimental import attempt_load
from utils.torch_utils import select_device
from utils.general import non_max_suppression, scale_coords
from utils.datasets import letterbox
from utils.plots import plot_one_box

app = Flask(__name__)

device = select_device('')
model = attempt_load('/home/jetson/yolov5/yolov5s_v5.pt', map_location=device)
model.eval()

gst = (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
    "nvvidconv flip-method=0 ! "
    "video/x-raw, width=416, height=416, format=BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=BGR ! appsink drop=1"
)

cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)

def generate_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break

        img = letterbox(frame, new_shape=416)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = img.copy()
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = model(img)[0]
        pred = non_max_suppression(pred, 0.25, 0.45)

        for det in pred:
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()

                for *xyxy, conf, cls in det:
                    label = "%s %.2f" % (model.names[int(cls)], conf)
                    plot_one_box(xyxy, frame, label=label, color=(0, 255, 0), line_thickness=2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(host='0.0.0.0', port=5000)
