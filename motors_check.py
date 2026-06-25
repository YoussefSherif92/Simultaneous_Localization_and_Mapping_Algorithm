import time
from jetracer.nvidia_racecar import NvidiaRacecar

car = NvidiaRacecar()


def reset_car():
    print("Resetting car...")
    car.throttle = 0.0
    car.steering = 0.0
    time.sleep(2)

reset_car()


try:
    while True:
        print("Forward")
        car.steering = 0.0
        car.throttle = 0.15
        time.sleep(2)

        print("Right")
        car.steering = 0.6
        car.throttle = 0.15
        time.sleep(1)

        print("Forward")
        car.steering = 0.0
        car.throttle = 0.15
        time.sleep(2)

        print("Left")
        car.steering = -0.6
        car.throttle = 0.15
        time.sleep(1)

        print("Backward")
        car.steering = 0.0
        car.throttle = -0.15
        time.sleep(1)

        print("Stop")
        car.throttle = 0.0
        car.steering = 0.0
        time.sleep(1)

except KeyboardInterrupt:
    car.throttle = 0.0
    car.steering = 0.0
    print("Stopped")
