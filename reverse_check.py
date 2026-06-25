import time
from jetracer.nvidia_racecar import NvidiaRacecar

car = NvidiaRacecar()

try:
    print("Stop")
    car.throttle = 0.0
    car.steering = 0.0
    time.sleep(2)

    print("Backward")
    car.throttle = -0.15
    time.sleep(3)

    print("Stop")
    car.throttle = 0.0
    time.sleep(1)

except KeyboardInterrupt:
    pass

finally:
    car.throttle = 0.0
    car.steering = 0.0
