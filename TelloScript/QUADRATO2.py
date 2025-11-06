import time
import threading
import datetime
import csv
import atexit
from djitellopy import Tello
import os

# === Initialization ===
tello = Tello()
position = [0.0, 0.0, 0.0]
velocity = [0.0, 0.0, 0.0]
last_time = time.time()
stop_flag = False

# === Log file ===
log_folder = r"D:\tello\Alessia"
os.makedirs(log_folder, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
imu_filename = os.path.join(log_folder, f"imu_log_{timestamp}.csv")
command_log_filename = os.path.join(log_folder, f"command_log_{timestamp}.txt")

imu_file = open(imu_filename, mode='w', newline='')
imu_writer = csv.writer(imu_file)
imu_writer.writerow(['Time', 'Ax', 'Ay', 'Az', 'Sx', 'Sy', 'Sz', 'x', 'y', 'z'])

# === Logging ===
def log_imu_data():
    global last_time
    try:
        now = time.time()
        dt = now - last_time
        last_time = now
        tello.get_current_state()
        ax = tello.get_acceleration_x()
        ay = tello.get_acceleration_y()
        az = tello.get_acceleration_z()
        sx = tello.get_speed_x()
        sy = tello.get_speed_y()
        sz = tello.get_speed_z()
        position[0] += sx * dt
        position[1] += sy * dt
        position[2] += sz * dt
        imu_writer.writerow([now, ax, ay, az, sx, sy, sz, *position])
    except Exception as e:
        print("❌ Errore IMU:", e)

def imu_loop():
    while not stop_flag:
        log_imu_data()
        time.sleep(0.1)

def log_command(cmd):
    with open(command_log_filename, 'a') as f:
        f.write(f"{datetime.datetime.now().strftime('%H:%M:%S')} - {cmd}\n")

# === Automatic Square Routine ===
def auto_mission():
    #tello.disable_mission_pads() # Disable Tello's mission pads. (Maybe?)
    try:
        log_command("takeoff")
        tello.takeoff()
        time.sleep(5)

        log_command("routine_quadrato")
        tello.go_xyz_speed(50, 0, 0, 10) # drone moves along x-axis by 50 cm at 10 cm/s speed. 
        time.sleep(10)

        tello.go_xyz_speed(0, 50, 0, 10) # drone moves along y-axis by 50 cm at 10 cm/s speed.
        time.sleep(10)

        tello.go_xyz_speed(-50, 0, 0, 10)
        time.sleep(10)

        tello.go_xyz_speed(0, -50, 0, 10)
        time.sleep(10)

        log_command("land")
        tello.land()
        time.sleep(2)
    except Exception as e:
        print("Errore di volo:", e)
        log_command(e)
    

# === Chiusura ===
def close_everything():
    try:
        imu_file.close()
        tello.end()
        print("🟢 Chiusura completata.")
    except Exception as e:
        print("Errore in chiusura:", e)

atexit.register(close_everything)

# === MAIN ===
if __name__ == "__main__":
    try:
        tello.connect()
        print(f"✅ Drone connesso. Batteria: {tello.get_battery()}%")
    except Exception as e:
        print("❌ Connessione fallita:", e)
        exit(1)

    imu_thread = threading.Thread(target=imu_loop)
    imu_thread.start()

    time.sleep(3)  # attesa iniziale prima della missione
    auto_mission()

    stop_flag = True
    imu_thread.join()