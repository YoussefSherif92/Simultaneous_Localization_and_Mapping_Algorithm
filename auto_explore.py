import time
from jetracer.nvidia_racecar import NvidiaRacecar

car = NvidiaRacecar()

def stop(t=1):
    print("Stop")
    car.throttle = 0.0
    car.steering = 0.0
    time.sleep(t)

def forward(speed=0.3, t=2):
    print("Forward")
    car.steering = 0.0
    car.throttle = speed
    time.sleep(t)
    stop()

def backward(speed=-0.3, t=2):
    print("Reverse arm")
    car.steering = 0.0
    car.throttle = -speed
    time.sleep(0.7)

    stop(0.7)

    print("Backward")
    car.throttle = -speed
    time.sleep(t)
    stop()

def left(speed=0.4, t=2):
    print("Turn Left")
    car.steering = -0.5
    car.throttle = speed
    time.sleep(t)
    stop()

def right(speed=0.4, t=2):
    print("Turn Right")
    car.steering = 0.5
    car.throttle = speed
    time.sleep(t)
    stop()

try:
    forward()
    backward()
    left()
    right()

finally:
    print("Final Stop")
    car.throttle = 0.0
    car.steering = 0.0
