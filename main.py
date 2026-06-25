import sys
sys.path.append('/home/jetson/yolov5')

import torch
from models.experimental import attempt_load
from utils.torch_utils import select_device

device = select_device('')
model = attempt_load('/home/jetson/yolov5/yolov5s_v5.pt', map_location=device)

print("YOLO loaded successfully")
print("Device:", device)
